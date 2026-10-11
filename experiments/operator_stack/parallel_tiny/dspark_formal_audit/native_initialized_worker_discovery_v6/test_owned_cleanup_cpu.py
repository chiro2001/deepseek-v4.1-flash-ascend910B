"""Linux CPU child lifecycle/pidfd tests only; no torch/SDK/NPU imports."""
import multiprocessing
import os
import signal
import unittest
from unittest.mock import patch

from owned_worker_cleanup import (require_pidfd_support,owner_identity,capture_owned_worker,finish_owned_workers)


def ignored_term_child(ready):
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    ready.set()
    while True:signal.pause()


def normal_cpu_child():
    return


@unittest.skipUnless(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'needs real Linux pidfd')
class OwnedCleanupCPU(unittest.TestCase):
    def setUp(self):
        self.owner=owner_identity();self.children=[]

    def child(self):
        context=multiprocessing.get_context('fork');ready=context.Event()
        process=context.Process(target=ignored_term_child,args=(ready,))
        process.start();self.children.append(process)
        self.assertTrue(ready.wait(2))
        return process,capture_owned_worker(process,0,self.owner)

    def tearDown(self):
        # Only these self-created test children; use the same verified pidfd path.
        for process in self.children:
            if process.exitcode is None:
                worker=capture_owned_worker(process,0,self.owner)
                report=finish_owned_workers([worker],self.owner,success=False,kill_wait_seconds=2)
                self.assertTrue(report['all_terminal'])
            process.join(timeout=0)

    def test_term_ignored_child_is_killed_by_pidfd_and_reaped(self):
        process,worker=self.child();process.terminate();process.join(timeout=.05)
        self.assertTrue(process.is_alive())
        report=finish_owned_workers([worker],self.owner,success=False,kill_wait_seconds=2)
        self.assertTrue(report['all_terminal']);self.assertEqual(report['pending_owned_pids'],[])
        self.assertEqual(report['worker_exit_codes'],{'0':-signal.SIGKILL})
        self.assertTrue(report['workers'][0]['hard_kill_sent']);self.assertFalse(process.is_alive())

    def test_successful_child_gets_normal_terminal_code_without_signal(self):
        process=multiprocessing.get_context('fork').Process(target=normal_cpu_child)
        process.start();self.children.append(process);worker=capture_owned_worker(process,0,self.owner)
        report=finish_owned_workers([worker],self.owner,success=True,success_grace_seconds=2,kill_wait_seconds=2)
        self.assertTrue(report['all_terminal']);self.assertEqual(report['worker_exit_codes'],{'0':0})
        self.assertFalse(report['workers'][0]['hard_kill_sent'])

    def assert_refused(self,worker,owner):
        report=finish_owned_workers([worker],owner,success=False,kill_wait_seconds=0)
        self.assertFalse(report['all_terminal']);self.assertEqual(report['worker_exit_codes'],{})
        self.assertEqual(report['pending_owned_pids'],[worker.process.pid])
        self.assertFalse(report['workers'][0]['hard_kill_sent'])
        self.assertIn('signal_refused_or_failed',report['workers'][0]);self.assertTrue(worker.process.is_alive())

    def test_changed_start_ticks_refuses_signal_without_pid_fallback(self):
        process,worker=self.child();worker.identity={**worker.identity,'start_ticks':worker.identity['start_ticks']+1}
        self.assert_refused(worker,self.owner)

    def test_wrong_parent_refuses_signal(self):
        process,worker=self.child();worker.identity={**worker.identity,'ppid':worker.identity['ppid']+1}
        self.assert_refused(worker,self.owner)

    def test_wrong_boot_refuses_signal(self):
        process,worker=self.child();worker.identity={**worker.identity,'boot_id':'CPU-only-wrong-boot'}
        self.assert_refused(worker,self.owner)

    def test_changed_driver_cmdline_refuses_signal(self):
        process,worker=self.child();owner={**self.owner,'cmdline_sha256':'0'*64}
        self.assert_refused(worker,owner)

    def test_missing_captured_birth_identity_refuses_signal(self):
        process,worker=self.child();worker.identity=None
        self.assert_refused(worker,self.owner)

    def test_kernel_pidfd_target_mismatch_refuses_signal(self):
        process,worker=self.child();original_fd=worker.pidfd
        worker.pidfd=os.pidfd_open(os.getpid(),0)
        try:self.assert_refused(worker,self.owner)
        finally:os.close(original_fd)

    def test_api_preflight_uses_only_null_signal_and_no_pid_fallback(self):
        with patch('owned_worker_cleanup.signal.pidfd_send_signal',wraps=signal.pidfd_send_signal) as sender:
            require_pidfd_support()
        self.assertEqual(sender.call_count,1);self.assertEqual(sender.call_args.args[1],0)


if __name__=='__main__':unittest.main()
