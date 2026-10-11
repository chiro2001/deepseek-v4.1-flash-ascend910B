"""Bounded schedule clock, separate from frontend async polling."""


class ProfileStepClock:
    def __init__(self, warmup, active, worker_driven):
        assert warmup >= 0 and active > 0
        self.warmup, self.active = warmup, active
        self.worker_driven = worker_driven
        self.limit = warmup + active + 1
        self.worker_calls = self.manual_calls = self.schedule_steps = 0

    def worker_step(self):
        self.worker_calls += 1
        advance = self.worker_driven and self.schedule_steps < self.limit
        if advance:
            self.schedule_steps += 1
        return advance

    def manual_step(self):
        assert not self.worker_driven, 'Async worker clock cannot be advanced by frontend RPC'
        assert self.schedule_steps < self.limit
        self.manual_calls += 1
        self.schedule_steps += 1

    def receipt(self):
        return {'driver': 'worker.execute_model' if self.worker_driven else 'manual RPC',
                'worker_calls': self.worker_calls, 'manual_calls': self.manual_calls,
                'schedule_steps': self.schedule_steps, 'required_schedule_steps': self.limit,
                'warmup': self.warmup, 'active': self.active,
                'complete': self.schedule_steps == self.limit}
