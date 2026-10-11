"""CPU read-only proof of an already mapped ABI object; no device API calls."""
import ctypes
import hashlib
import os
from pathlib import Path


class ProofUnavailable(RuntimeError):
    def __init__(self, message, proof=None):
        super().__init__(message)
        self.loaded_abi_proof=proof


def parse_maps(text):
    entries=[]
    for line in text.splitlines():
        parts=line.split(None,5)
        if len(parts)!=6 or not parts[5].startswith('/'):
            continue
        start,end=(int(value,16) for value in parts[0].split('-'))
        major,minor=(int(value,16) for value in parts[3].split(':'))
        entries.append({'range':parts[0],'start':start,'end':end,'permissions':parts[1],
                        'file_offset':int(parts[2],16),'maps_dev':[major,minor],
                        'inode':int(parts[4]),'raw_path':parts[5],
                        'deleted':parts[5].endswith(' (deleted)')})
    return entries


def parse_mountinfo(text):
    mounts=[]
    for line in text.splitlines():
        fields=line.split();split=fields.index('-')
        mounts.append({'mount_id':int(fields[0]),'parent_mount_id':int(fields[1]),
                       'root':fields[3],'device':[int(value) for value in fields[2].split(':')],
                       'mount_point':fields[4].replace('\\040',' '),
                       'filesystem':fields[split+1],'source':fields[split+2],
                       'mountinfo_line':line})
    return mounts


def stat_metadata(stat):
    return {'inode':stat.st_ino,'dev':[os.major(stat.st_dev),os.minor(stat.st_dev)],'bytes':stat.st_size}


def _full_fd_sha(fd):
    digest=hashlib.sha256();offset=0
    while True:
        data=os.pread(fd,1024*1024,offset)
        if not data:break
        digest.update(data);offset+=len(data)
    return digest.hexdigest(),offset


def _read_exact(fd, address, size):
    chunks=[];remaining=size
    while remaining:
        chunk=os.pread(fd,min(remaining,1024*1024),address)
        if not chunk:raise ProofUnavailable('Short read of actual mapped executable memory')
        chunks.append(chunk);address+=len(chunk);remaining-=len(chunk)
    return b''.join(chunks)


def verify_proof(proof, expected_sha256, required_symbols):
    segments=proof['segments'];mapped=proof['mapped_file_stat']
    if not segments or any(segment['deleted'] for segment in segments):
        raise ValueError('Missing/deleted actual library mapping')
    identity={(tuple(segment['maps_dev']),segment['inode']) for segment in segments}
    if len(identity)!=1 or mapped['inode']!=segments[0]['inode']:
        raise ValueError('Kernel map_files inode differs from the actual mapped object')
    if proof['current_path_stat']!=mapped:
        raise ValueError('Current ABI path inode/device/extent differs from kernel map_files')
    resolved=proof['resolved_path']
    if not Path(resolved).is_absolute() or proof['map_files_link'].endswith(' (deleted)'):
        raise ValueError('Actual mapped ABI path is missing or deleted')
    if str(Path(proof['map_files_link']).resolve())!=resolved:
        raise ValueError('Kernel map_files link differs from the exact ABI path')
    if proof['mapped_file_sha256']!=expected_sha256 or proof['mapped_file_bytes']!=mapped['bytes']:
        raise ValueError('Actual kernel-mapped object full content SHA/extent differs')
    if mapped['dev']!=segments[0]['maps_dev']:
        overlay=proof.get('overlay_mount')
        if (not overlay or overlay['filesystem']!='overlay' or overlay['device']!=mapped['dev']
                or not overlay.get('mountinfo_line')
                or not (resolved==overlay['mount_point']
                        or resolved.startswith(overlay['mount_point'].rstrip('/')+'/'))):
            raise ValueError('Distinct device views need explicit matching overlay mount provenance')
    executable=[segment for segment in segments if 'x' in segment['permissions']]
    if any(segment['permissions'][:3]!='r-x' for segment in executable):
        raise ValueError('Actual ABI executable mappings must have read/execute-only permissions')
    comparisons=proof.get('executable_memory',[])
    if (len(comparisons)!=len(executable) or not executable
            or {item['range'] for item in comparisons}!={segment['range'] for segment in executable}):
        raise ValueError('Missing actual executable mapped-segment content proof')
    for item in comparisons:
        segment=next(segment for segment in executable if segment['range']==item['range'])
        if (item['bytes']!=segment['end']-segment['start'] or item['file_offset']!=segment['file_offset']
                or item['file_offset']<0 or item['file_offset']+item['bytes']>mapped['bytes']
                or item['memory_sha256']!=item['mapped_file_segment_sha256'] or item['byte_equal'] is not True):
            raise ValueError('Actual executable memory differs from the frozen mapped ELF bytes')
    if set(proof.get('symbol_addresses',{}))!=set(required_symbols):
        raise ValueError('Required NOLOAD ABI symbols are missing')
    for symbol,address in proof['symbol_addresses'].items():
        if type(address) is not int or address<=0 or not any(segment['start']<=address<segment['end'] for segment in executable):
            raise ValueError('NOLOAD symbol address is outside this actual object executable segments: '+symbol)
    return {'loaded_object_identity_proven':True,'device_audit_passed':False,
            'safe_formal_receipt':False,'formal_model_validated':False}


def prove_existing_mapped_object(path, expected_sha256, required_symbols):
    """Open self/map_files, compare self/mem, NOLOAD and take symbol addresses.

    No fallback to current-path SHA alone, no normal dlopen, no symbol call.
    The retained handle may subsequently be used by a separately authorized
    initialized-worker gate to query its already-existing context/mapping.
    """
    if not hasattr(os,'RTLD_NOLOAD'):
        raise ProofUnavailable('This process cannot use RTLD_NOLOAD')
    requested=Path(path);resolved=requested.resolve()
    maps_text=Path('/proc/self/maps').read_text()
    segments=[entry for entry in parse_maps(maps_text)
              if Path(entry['raw_path'].removesuffix(' (deleted)')).resolve()==resolved]
    proof={'pid':os.getpid(),'requested_path':str(requested),'resolved_path':str(resolved),
           'segments':segments,'device_api_calls':0,'new_library_loads':0}
    if not segments:raise ProofUnavailable('Requested ABI path is absent from actual self/maps',proof)
    if any(entry['deleted'] for entry in segments):
        raise ProofUnavailable('Requested ABI has a deleted actual self/maps segment',proof)
    if len({(tuple(entry['maps_dev']),entry['inode']) for entry in segments})!=1:
        raise ProofUnavailable('Requested ABI path resolves to ambiguous actual mapping identities',proof)
    map_file=Path('/proc/self/map_files')/segments[0]['range']
    proof['map_files_path']=str(map_file)
    fd=mem_fd=None
    try:
        try:fd=os.open(map_file,os.O_RDONLY|os.O_CLOEXEC)
        except OSError as exc:raise ProofUnavailable('Cannot open actual self/map_files object: '+str(exc),proof) from exc
        mapped_stat=stat_metadata(os.fstat(fd))
        proof['mapped_file_stat']=mapped_stat
        mapped_sha,mapped_bytes=_full_fd_sha(fd)
        proof.update(mapped_file_sha256=mapped_sha,mapped_file_bytes=mapped_bytes)
        mountinfo_text=Path('/proc/self/mountinfo').read_text()
        mounts=parse_mountinfo(mountinfo_text)
        applicable=[mount for mount in mounts
                    if str(resolved)==mount['mount_point'] or str(resolved).startswith(mount['mount_point'].rstrip('/')+'/')]
        overlay=max(applicable,key=lambda mount:len(mount['mount_point'])) if applicable else None
        proof.update(map_files_link=os.readlink(map_file),current_path_stat=stat_metadata(resolved.stat()),
                     mountinfo_sha256=hashlib.sha256(mountinfo_text.encode()).hexdigest(),
                     overlay_mount=overlay,device_views_differ=mapped_stat['dev']!=segments[0]['maps_dev'],
                     executable_memory=[],symbol_addresses={})
        try:mem_fd=os.open('/proc/self/mem',os.O_RDONLY|os.O_CLOEXEC)
        except OSError as exc:raise ProofUnavailable('Cannot read actual self/mem: '+str(exc),proof) from exc
        for segment in segments:
            if 'x' not in segment['permissions']:continue
            size=segment['end']-segment['start']
            actual=_read_exact(mem_fd,segment['start'],size)
            frozen=_read_exact(fd,segment['file_offset'],size)
            proof['executable_memory'].append({'range':segment['range'],'bytes':size,
                'file_offset':segment['file_offset'],'memory_sha256':hashlib.sha256(actual).hexdigest(),
                'mapped_file_segment_sha256':hashlib.sha256(frozen).hexdigest(),'byte_equal':actual==frozen})
        # Take addresses only. Do not invoke any ACL/runtime API in discovery.
        handle=ctypes.CDLL(str(resolved),mode=os.RTLD_NOW|os.RTLD_NOLOAD)
        for symbol in required_symbols:
            function=getattr(handle,symbol)
            proof['symbol_addresses'][symbol]=ctypes.cast(function,ctypes.c_void_p).value
        verify_proof(proof,expected_sha256,required_symbols)
        return proof,handle
    except OSError as exc:
        raise ProofUnavailable('Actual mapping/memory proof is unavailable: '+str(exc),proof) from exc
    except Exception as exc:
        if getattr(exc,'loaded_abi_proof',None) is None:exc.loaded_abi_proof=proof
        raise
    finally:
        if mem_fd is not None:os.close(mem_fd)
        if fd is not None:os.close(fd)
