"""CPU false-positive contracts. No torch, device call, or actual library load."""
import copy
from contextlib import contextmanager,ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mapped_object_proof import verify_proof,ProofUnavailable,_read_exact,prove_existing_mapped_object
from controller_cpu import write_controller_bundle
from contracts import validate_controller,read_frozen_json
from runtime_scope import ABI_SYMBOLS,AlreadyLoadedRuntime


class ProofCPU(unittest.TestCase):
    def setUp(self):
        self.sha='a'*64
        self.proof={'pid':os.getpid(),'resolved_path':'/CPU-only/lib.so','map_files_link':'/CPU-only/lib.so',
            'segments':[{'range':'1000-2000','start':4096,'end':8192,'permissions':'r-xp',
            'file_offset':0,'maps_dev':[253,2],'inode':123,'raw_path':'/CPU-only/lib.so','deleted':False}],
            'mapped_file_stat':{'inode':123,'dev':[0,259],'bytes':4096},
            'current_path_stat':{'inode':123,'dev':[0,259],'bytes':4096},
            'mapped_file_sha256':self.sha,'mapped_file_bytes':4096,
            'overlay_mount':{'device':[0,259],'filesystem':'overlay','mount_point':'/',
                             'mountinfo_line':'1 0 0:259 / / rw - overlay overlay rw'},
            'executable_memory':[{'range':'1000-2000','file_offset':0,'bytes':4096,
               'memory_sha256':'b'*64,'mapped_file_segment_sha256':'b'*64,'byte_equal':True}],
            'symbol_addresses':{'aclrtGetCurrentContext':4100,'aclrtGetDevice':4110}}
        self.symbols=tuple(self.proof['symbol_addresses'])

    def check(self):return verify_proof(self.proof,self.sha,self.symbols)

    def test_overlay_needs_content_and_symbol_proof_not_just_removed_dev_check(self):
        result=self.check();self.assertFalse(result['device_audit_passed'])
        self.assertFalse(result['safe_formal_receipt']);self.assertFalse(result['formal_model_validated'])

    def test_wrong_inode_rejected(self):
        self.proof['mapped_file_stat']['inode']=124
        with self.assertRaisesRegex(ValueError,'inode'):self.check()

    def test_mapped_file_content_change_rejected(self):
        self.proof['mapped_file_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'full content'):self.check()

    def test_runtime_executable_patch_rejected(self):
        self.proof['executable_memory'][0]['memory_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'executable memory'):self.check()

    def test_symbol_belongs_to_other_library_or_data_segment_rejected(self):
        for address in (0,9000,True):
            self.proof['symbol_addresses']['aclrtGetDevice']=address
            with self.assertRaisesRegex(ValueError,'symbol address'):self.check()

    def test_device_view_mismatch_without_overlay_provenance_rejected(self):
        self.proof['overlay_mount']['filesystem']='ext4'
        with self.assertRaisesRegex(ValueError,'overlay mount'):self.check()

    def test_missing_segment_memory_or_symbol_rejected(self):
        old=copy.deepcopy(self.proof)
        self.proof['executable_memory']=[]
        with self.assertRaisesRegex(ValueError,'content proof'):self.check()
        self.proof=old;self.proof['symbol_addresses'].pop('aclrtGetDevice')
        with self.assertRaisesRegex(ValueError,'symbols'):self.check()

    def test_mem_short_read_or_permission_failure_cannot_fallback(self):
        with patch('mapped_object_proof.os.pread',return_value=b''):
            with self.assertRaisesRegex(ProofUnavailable,'Short read'):_read_exact(1,4096,4096)
        with patch('mapped_object_proof.os.pread',side_effect=PermissionError('CPU-only denied')):
            with self.assertRaises(PermissionError):_read_exact(1,4096,4096)

    def test_map_files_permission_failure_stops_before_mem_or_noload(self):
        maps='1000-2000 r-xp 00000000 fd:02 123 /CPU-only/lib.so\n'
        with patch('mapped_object_proof.Path.read_text',return_value=maps), \
             patch('mapped_object_proof.os.open',side_effect=PermissionError('CPU-only denied')), \
             patch('mapped_object_proof.ctypes.CDLL') as loader:
            with self.assertRaisesRegex(ProofUnavailable,'map_files'):
                prove_existing_mapped_object('/CPU-only/lib.so',self.sha,self.symbols)
            loader.assert_not_called()

    def test_wrong_current_path_inode_is_rejected(self):
        self.proof['current_path_stat']['inode']=124
        with self.assertRaisesRegex(ValueError,'Current ABI path'):self.check()

    def test_deleted_or_wrong_kernel_link_is_rejected(self):
        for link in ('/CPU-only/other.so','/CPU-only/lib.so (deleted)'):
            self.proof['map_files_link']=link
            with self.assertRaisesRegex(ValueError,'ABI path'):self.check()

    def test_overlay_mount_must_cover_exact_library_path(self):
        self.proof['overlay_mount']['mount_point']='/unrelated'
        with self.assertRaisesRegex(ValueError,'overlay mount'):self.check()

    def test_executable_write_permissions_duplicate_proof_or_wrong_offset_rejected(self):
        old=copy.deepcopy(self.proof)
        self.proof['segments'][0]['permissions']='rwxp'
        with self.assertRaisesRegex(ValueError,'permissions'):self.check()
        self.proof=copy.deepcopy(old);self.proof['executable_memory']*=2
        with self.assertRaisesRegex(ValueError,'content proof'):self.check()
        self.proof=copy.deepcopy(old);self.proof['executable_memory'][0]['file_offset']=1
        with self.assertRaisesRegex(ValueError,'executable memory'):self.check()

    def test_runtime_does_not_use_noload_or_device_symbols_when_proof_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'CPU-only-lib';path.write_bytes(b'CPU-only')
            entry={'path':str(path),'bytes':8,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch('runtime_scope.prove_existing_mapped_object',side_effect=ProofUnavailable('CPU-only bad executable')) as prover, \
                 patch('runtime_scope.ctypes.CDLL') as loader:
                with self.assertRaisesRegex(ProofUnavailable,'bad executable'):
                    AlreadyLoadedRuntime({'acl':entry,'runtime':entry})
                self.assertEqual(prover.call_count,1);loader.assert_not_called()

    def descriptions(self):
        libraries={name:{'path':'/CPU-only/lib.so','bytes':4096,'sha256':self.sha} for name in ABI_SYMBOLS}
        result=[]
        for rank in range(8):
            proofs={}
            for name,symbols in ABI_SYMBOLS.items():
                proof=copy.deepcopy(self.proof);proof['pid']=os.getpid()+rank
                # Real processes use independent ASLR addresses, not common source identity.
                offset=rank*65536
                proof['segments'][0].update(start=4096+offset,end=8192+offset,range=f'{4096+offset:x}-{8192+offset:x}')
                proof['executable_memory'][0]['range']=proof['segments'][0]['range']
                proof['symbol_addresses']={symbol:4100+offset+index for index,symbol in enumerate(symbols)}
                proofs[name]=proof
            result.append({'tp_rank':rank,'pid':os.getpid()+rank,'boot_id':'CPU-only-boot',
                'normal_init_return_ns':rank+1,'source_manifest':{'libraries':libraries},
                'loaded_abi_proofs':proofs,'device_audit_passed':False,'safe_formal_receipt':False})
        return result

    def test_rank_aslr_proofs_remain_separate_and_bound_to_common_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            job=Path(folder)/'CPU-only-job';gate=job/'gate'
            config=Path(__file__).parent/'evidence/checkpoint_config/config.json'
            descriptions=self.descriptions()
            receipt=write_controller_bundle(descriptions,metadata_root=gate/'metadata',
                output_root=gate/'sdk_results',checkpoint_config_path=config,
                audit_job_id=job.name,nonce='CPU-only-nonce',variant='static_default',physical_chips=list(range(8,16)))
            controller=read_frozen_json(receipt['controller_path'],receipt['controller_sha256'])
            self.assertFalse(validate_controller(controller)['device_audit_passed'])
            manifest=read_frozen_json(controller['source_manifest_path'],controller['source_manifest_sha256'])
            self.assertNotIn('pid',manifest);self.assertNotIn('loaded_abi_proofs',manifest)
            for item in controller['worker_processes']:
                proofs=read_frozen_json(item['loaded_abi_proofs_path'],item['loaded_abi_proofs_sha256'])
                self.assertEqual(proofs,descriptions[item['tp_rank']]['loaded_abi_proofs'])
                self.assertEqual(proofs['acl']['pid'],item['pid'])
            controller['audit_job_id']='gate'
            with self.assertRaisesRegex(ValueError,'not bound'):validate_controller(controller)

    def test_controller_rejects_missing_or_other_pid_mapped_proof_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            job=Path(folder)/'CPU-only-job';gate=job/'gate'
            kwargs={'metadata_root':gate/'metadata','output_root':gate/'sdk_results',
                'checkpoint_config_path':Path(__file__).parent/'evidence/checkpoint_config/config.json',
                'audit_job_id':job.name,'nonce':'CPU-only-nonce','variant':'static_default','physical_chips':list(range(8,16))}
            descriptions=self.descriptions();descriptions[0]['loaded_abi_proofs']['acl']['pid']+=100
            with self.assertRaisesRegex(ValueError,'new rank PID'):write_controller_bundle(descriptions,**kwargs)
            descriptions=self.descriptions();descriptions[0]['loaded_abi_proofs'].pop('runtime')
            with self.assertRaisesRegex(ValueError,'complete mapped ABI'):write_controller_bundle(descriptions,**kwargs)
            self.assertFalse(kwargs['metadata_root'].exists())

    @contextmanager
    def mocked_mapping(self, *, mem_permission=False, memory=None, symbol_address=4100):
        """CPU fixture for the complete read-only syscall sequence, not a device receipt."""
        data=b'\x7fELF'+b'\0'*(4096-4)+b'A'*4096
        maps='1000-2000 r-xp 00001000 fd:02 123 /CPU-only/lib.so\n'
        mount='1 0 0:259 / / rw - overlay overlay rw\n'
        stat=SimpleNamespace(st_ino=123,st_dev=os.makedev(0,259),st_size=len(data))
        def read_text(path):
            if str(path)=='/proc/self/maps':return maps
            if str(path)=='/proc/self/mountinfo':return mount
            raise AssertionError('Unexpected read: '+str(path))
        def open_fd(path,flags):
            if str(path)=='/proc/self/map_files/1000-2000':return 10
            if str(path)=='/proc/self/mem':
                if mem_permission:raise PermissionError('CPU-only self/mem denied')
                return 11
            raise AssertionError('Unexpected open: '+str(path))
        def pread(fd,size,offset):
            if fd==10:return data[offset:offset+size]
            if fd==11:return (b'A'*4096 if memory is None else memory)[:size]
            raise AssertionError('Unexpected fd')
        with ExitStack() as stack:
            stack.enter_context(patch('mapped_object_proof.Path.read_text',autospec=True,side_effect=read_text))
            stack.enter_context(patch('mapped_object_proof.Path.stat',return_value=stat))
            stack.enter_context(patch('mapped_object_proof.os.open',side_effect=open_fd))
            stack.enter_context(patch('mapped_object_proof.os.fstat',return_value=stat))
            stack.enter_context(patch('mapped_object_proof.os.pread',side_effect=pread))
            stack.enter_context(patch('mapped_object_proof.os.readlink',return_value='/CPU-only/lib.so'))
            stack.enter_context(patch('mapped_object_proof.os.close'))
            stack.enter_context(patch('mapped_object_proof.ctypes.cast',return_value=SimpleNamespace(value=symbol_address)))
            loader=stack.enter_context(patch('mapped_object_proof.ctypes.CDLL',return_value=SimpleNamespace(
                aclrtGetCurrentContext=object(),aclrtGetDevice=object())))
            yield loader,hashlib.sha256(data).hexdigest()

    def test_complete_prover_reads_kernel_fd_and_live_memory_at_actual_file_offset(self):
        with self.mocked_mapping() as (loader,sha):
            proof,handle=prove_existing_mapped_object('/CPU-only/lib.so',sha,self.symbols)
            self.assertEqual(proof['mapped_file_bytes'],8192)
            self.assertEqual(proof['executable_memory'][0]['file_offset'],4096)
            self.assertTrue(proof['executable_memory'][0]['byte_equal'])
            self.assertEqual(proof['device_api_calls'],0);self.assertEqual(proof['new_library_loads'],0)
            self.assertIs(handle,loader.return_value)
            loader.assert_called_once_with('/CPU-only/lib.so',mode=os.RTLD_NOW|os.RTLD_NOLOAD)
            self.assertFalse(verify_proof(proof,sha,self.symbols)['device_audit_passed'])

    def test_complete_prover_mem_permission_failure_cannot_use_noload_fallback(self):
        with self.mocked_mapping(mem_permission=True) as (loader,sha):
            with self.assertRaisesRegex(ProofUnavailable,'self/mem') as rejected:
                prove_existing_mapped_object('/CPU-only/lib.so',sha,self.symbols)
            proof=rejected.exception.loaded_abi_proof
            self.assertEqual(proof['segments'][0]['maps_dev'],[253,2])
            self.assertEqual(proof['mapped_file_stat']['dev'],[0,259])
            self.assertEqual(proof['mapped_file_sha256'],sha)
            loader.assert_not_called()

    def test_complete_prover_rejects_changed_live_code_and_wrong_symbol_segment(self):
        with self.mocked_mapping(memory=b'B'*4096) as (_,sha):
            with self.assertRaisesRegex(ValueError,'executable memory') as rejected:
                prove_existing_mapped_object('/CPU-only/lib.so',sha,self.symbols)
            self.assertFalse(rejected.exception.loaded_abi_proof['executable_memory'][0]['byte_equal'])
        with self.mocked_mapping(symbol_address=9000) as (_,sha):
            with self.assertRaisesRegex(ValueError,'symbol address'):
                prove_existing_mapped_object('/CPU-only/lib.so',sha,self.symbols)


if __name__=='__main__':unittest.main()
