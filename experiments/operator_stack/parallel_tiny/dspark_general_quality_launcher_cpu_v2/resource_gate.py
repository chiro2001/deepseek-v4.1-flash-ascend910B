"""Pure fresh-occupancy/health guards. Reading npu-smi is a future launch step."""
import re

CHIPS = tuple(range(8, 16))


def resources(raw):
    devices, addresses = {}, {}
    board = health = None
    for line in raw.splitlines():
        match = re.match(r'^\|\s*(\d+)\s+Ascend\S*\s*\|\s*(\S+)', line)
        if match:
            board, health = int(match[1]), match[2]
        physical = re.match(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*[0-9A-Fa-f]{4}:', line)
        if physical:
            chip, physical_id = map(int, physical.groups())
            if board is None or health is None or physical_id in devices:
                raise ValueError('Ambiguous npu-smi physical device table')
            devices[physical_id] = {'health': health, 'processes': []}
            addresses[board, chip] = physical_id
        process = re.match(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(\S+)', line)
        if process:
            process_board, chip, pid = map(int, process.groups()[:3])
            if (process_board, chip) not in addresses:
                raise ValueError('Process table has no physical device owner')
            devices[addresses[process_board, chip]]['processes'].append({'pid': pid, 'name': process[4]})
    if not devices:
        raise ValueError('Cannot parse npu-smi; allocation is refused')
    return devices


def verify(devices, health_details, allow_known_alarm=False, previous=None):
    if not set(CHIPS).issubset(devices):
        raise ValueError('All eight requested physical chips must be present')
    for chip in CHIPS:
        row = devices[chip]
        if row['processes']:
            raise ValueError('Requested chip is occupied: ' + str(chip))
        allowed = ('OK', 'Alarm') if allow_known_alarm else ('OK',)
        if row['health'] not in allowed:
            raise ValueError('Chip health is outside this allocation: ' + str(chip))
        codes = [line.split(':', 1)[1].strip() for line in health_details[chip].splitlines()
                 if 'Error Code' in line and ':' in line]
        expected = ['80C98001'] if row['health'] == 'Alarm' else ['NA']
        if codes != expected:
            raise ValueError('Unexpected physical chip health code: ' + str(chip))
        if previous is not None and row['health'] != previous[chip]['health']:
            raise ValueError('Chip health changed before launch: ' + str(chip))
    return True
