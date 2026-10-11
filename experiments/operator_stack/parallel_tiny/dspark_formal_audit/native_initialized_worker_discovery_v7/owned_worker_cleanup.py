"""Only reap this driver's captured children; never scan or signal other jobs."""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import signal
import time


def require_pidfd_support():
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise RuntimeError('Owned worker cleanup requires Linux pidfd APIs; no numeric-PID fallback')
    fd=os.pidfd_open(os.getpid(),0)
    try:signal.pidfd_send_signal(fd,0,None,0)  # Permission/API check; no delivered signal.
    finally:os.close(fd)


def read_process_identity(pid):
    text=Path('/proc/'+str(pid)+'/stat').read_text()
    fields=text[text.rfind(')')+2:].split()
    return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def owner_identity():
    result=read_process_identity(os.getpid())
    result['cmdline_sha256']=hashlib.sha256(Path('/proc/self/cmdline').read_bytes()).hexdigest()
    return result


@dataclass
class OwnedWorker:
    rank: int
    process: object
    pidfd: int | None
    identity: dict | None


def capture_owned_worker(process, rank, owner):
    require_pidfd_support()
    if owner['pid']!=os.getpid() or process.pid is None:
        raise ValueError('Capture only a child created by this actual driver')
    try:fd=os.pidfd_open(process.pid,0)
    except ProcessLookupError:
        process.join(timeout=1)
        if process.exitcode is None:raise RuntimeError('New owned child disappeared without a terminal exit code')
        return OwnedWorker(rank,process,None,None)
    try:
        try:identity=read_process_identity(process.pid)
        except FileNotFoundError:
            process.join(timeout=1)
            if process.exitcode is None:raise RuntimeError('Owned child identity unavailable without terminal exit code')
            return OwnedWorker(rank,process,fd,None)
        if identity['ppid']!=owner['pid'] or identity['boot_id']!=owner['boot_id']:
            raise ValueError('New child PID/PPID/boot identity differs from this driver')
        return OwnedWorker(rank,process,fd,identity)
    except BaseException:
        os.close(fd)
        raise


def _verify_signal_target(worker, owner):
    current_owner=owner_identity()
    if any(current_owner[key]!=owner[key] for key in ('pid','start_ticks','boot_id','cmdline_sha256')):
        raise ValueError('Actual driver PID/boot/start/cmdline identity changed')
    if worker.pidfd is None or worker.identity is None:
        raise ValueError('Cannot signal a child without its captured pidfd and birth identity')
    actual=read_process_identity(worker.process.pid)
    if (any(actual[key]!=worker.identity[key] for key in ('pid','ppid','start_ticks','boot_id'))
            or actual['ppid']!=owner['pid']):
        raise ValueError('Refuse signal: captured child PID/PPID/boot/start identity differs')
    fields=dict(line.split(':',1) for line in Path('/proc/self/fdinfo/'+str(worker.pidfd)).read_text().splitlines()
                if ':' in line)
    target=int(fields.get('Pid','-2').strip())
    if target==-1:raise ProcessLookupError('Captured pidfd target already exited')
    if target!=worker.process.pid:
        raise ValueError('Refuse signal: kernel pidfd target differs from the captured child PID')


def finish_owned_workers(workers, owner, *, success, success_grace_seconds=5, kill_wait_seconds=10):
    """On failure bypass SDK SIGTERM handlers with verified pidfd SIGKILL.

    Successful workers first get bounded normal shutdown time. All codes are
    actual reaped Process exitcodes. An incomplete cleanup explicitly reports
    pending PIDs; the driver must not enter multiprocessing's implicit join.
    """
    report={'owner':owner,'scope':'captured driver-owned worker processes only',
            'signal':'SIGKILL via pidfd after identity verification',
            'workers':[],'worker_exit_codes':{},'all_terminal':False,'pending_owned_pids':[]}
    if success:
        deadline=time.monotonic()+success_grace_seconds
        for worker in workers:worker.process.join(timeout=max(0,deadline-time.monotonic()))
    for worker in workers:
        process=worker.process;process.join(timeout=0)
        row={'rank':worker.rank,'pid':process.pid,'birth_identity':worker.identity,'hard_kill_sent':False}
        report['workers'].append(row)
        if process.exitcode is not None:continue
        try:
            _verify_signal_target(worker,owner)
            signal.pidfd_send_signal(worker.pidfd,signal.SIGKILL,None,0)
            row['hard_kill_sent']=True
        except (ProcessLookupError,FileNotFoundError):
            # A concurrently exited captured child is reaped below; no fallback.
            row['exited_during_identity_or_signal']=True
        except BaseException as exc:
            row['signal_refused_or_failed']=type(exc).__name__+': '+str(exc)
    deadline=time.monotonic()+kill_wait_seconds
    for worker,row in zip(workers,report['workers']):
        worker.process.join(timeout=max(0,deadline-time.monotonic()))
        code=worker.process.exitcode
        row['terminal']=code is not None
        if code is None:report['pending_owned_pids'].append(worker.process.pid)
        else:
            row['exitcode']=code
            report['worker_exit_codes'][str(worker.rank)]=code
        if worker.pidfd is not None:
            os.close(worker.pidfd);worker.pidfd=None
    report['all_terminal']=not report['pending_owned_pids']
    return report
