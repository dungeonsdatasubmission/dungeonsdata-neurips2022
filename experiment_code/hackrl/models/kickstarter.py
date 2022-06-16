import torch


class KickStarter(torch.nn.Module):
    def __init__(self, student, teacher, always_run_teacher=True, run_teacher_hs=False):
        super(KickStarter, self).__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        self._k = "kick_"
        self.always_run_teacher = always_run_teacher
        self.run_teacher_hs = run_teacher_hs

    @property
    def version(self):
        return self.student.version

    @version.setter
    def version(self, v):
        self.student.version = v

    def initial_state(self, *args, **kwargs):
        return (
            self.student.initial_state(*args, **kwargs),
            self.teacher.initial_state(*args, **kwargs),
        )

    def forward(self, inputs, core_state):
        student_core, teacher_core = core_state
        if self.run_teacher_hs:
            student_core = teacher_core
        student_outputs, student_core = self.student(inputs, student_core)
        if self.always_run_teacher or not self.training:
            with torch.no_grad():
                teacher_outputs, teacher_core = self.teacher(inputs, teacher_core)
            for t in teacher_outputs:
                student_outputs[self._k + t] = teacher_outputs[t]
        return student_outputs, (student_core, teacher_core)
