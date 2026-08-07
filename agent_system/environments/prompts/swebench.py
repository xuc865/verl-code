# Copyright 2026 The DIDPO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- Coding agent (ReAct-style) --------------------- #
# ReAct (Yao et al., 2022): each step = Thought → Action; env returns Observation.
# Thought uses <think> (parsed by projection.py). Action is exactly one
# XML block below. Act may inspect, edit, run tests, or finish — not always edit.

_ACTION_SPEC_COMMON = """## ReAct loop (repeat each step until solved)
Thought: Reason about the task, prior observations, and your next move.
Action: Exactly ONE command below (inspect, edit, run tests, or submit).
Observation: Returned by the environment after your action (shown on the next step).

## Available actions (pick exactly one per step)
- Inspect listing:         <execute_bash>ls -la</execute_bash>
- Inspect a source file:   <execute_bash>cat solution.py</execute_bash>
{run_program_bullet}- Overwrite/create a file: <edit path="solution.py"><code>
# the FULL new content of the file
def foo(a, b):
    return a + b
</code></edit>
- Patch a unique region: <edit path="solution.py"><search>old lines</search><replace>new lines</replace></edit>
- Append a fragment: <edit path="solution.py"><insert>snippet</insert></edit>
- Patch a unique region (preferred for small fixes):
  <edit path="solution.py"><search>return 0</search><replace>return a + b</replace></edit>
- Append a fragment: <edit path="solution.py"><insert>
# helper added at end of file
</insert></edit>
- Submit current solution: <finish></finish>

## Rules
- Every step MUST be: Thought inside <think>...</think>, then one Action block.
- Do NOT output Observation yourself — the environment provides it.
- Progress the task: use Thought to decide whether to inspect, edit, or finish.
  Avoid repeating the same Action when the last Observation already gave that information.
- Allowed actions only: <edit> (overwrite / patch / insert), <execute_bash> for
  ls / cat / running your program or tests, and <finish>. Do NOT use grep/rg/ag
  or other search-in-file shell tools — inspect with cat, then edit.
{mode_rules}
- Grading runs the full hidden suite only at submission (<finish> or turn limit).
"""

_RUN_PROGRAM_BULLET = """- Run your program on sample input you provide (stdout/stderr only; no expected answers):
                           <execute_bash>echo '1 2' | python solution.py</execute_bash>
"""

_RULES_BLIND = """- Hidden unit tests are NOT executed during repair — you do not see pass/fail or expected outputs.
  Reason from the problem statement and your own inspection; submit with <finish> when ready."""

_RULES_EXEC = """- You may run your program with stdin you choose (see action above). You see stdout/stderr/traceback only.
  Hidden grading suite and pytest are NOT run during repair — no pass/fail vs expected answers.
- If your self-test input matches a hidden case and output is correct, you earn step reward immediately.
- Otherwise a clean run (exit 0) earns a small step reward; logic is fully graded at <finish>."""

_RULES_INTERACTIVE = """- After each <edit>, the environment automatically runs preset hidden test cases from the
  dataset (stdin/stdout pairs bundled with the problem — not tests you invent).
- You may also trigger them with <execute_bash>python solution.py</execute_bash>; feedback shows
  pass/fail counts and expected vs got for the first failing preset case.
- Step reward scales with the fraction of preset cases passed; full solve at <finish> or when all pass."""

_ACTION_SPEC_TAIL = """
## One-step example
<think>
I should read the current stub before changing it.
</think>
<execute_bash>cat solution.py</execute_bash>

## Multi-turn example (how Action/Observation trace looks across steps)
Action 1: <think>Read the stub first.</think>
<execute_bash>cat solution.py</execute_bash>
Observation 1: def solve(): pass  # TODO

Action 2: <think>Fix the buggy line only.</think>
<edit path="solution.py"><search>return 0</search><replace>return a + b</replace></edit>
Observation 2: Patched solution.py (...)

Action 3 (alternative — full rewrite): <think>Rewrite the file.</think>
<edit path="solution.py"><code>
def solve():
    return 42
</code></edit>
Observation 3: Edited solution.py (...)

Action 4: <think>Re-read the stub and align with the problem statement.</think>
<execute_bash>cat solution.py</execute_bash>
Observation 4: def solve(): ...

Action 5: <think>Implementation looks correct; submit for hidden grading.</think>
<finish></finish>

Your next output is only Thought + Action for the current step (not the full trace).
"""


def _with_examples(base: str) -> str:
    return base + _ACTION_SPEC_TAIL


_ACTION_SPEC_BLIND = _with_examples(_ACTION_SPEC_COMMON.format(
    run_program_bullet="",
    mode_rules=_RULES_BLIND,
))
_ACTION_SPEC_EXEC = _with_examples(_ACTION_SPEC_COMMON.format(
    run_program_bullet=_RUN_PROGRAM_BULLET,
    mode_rules=_RULES_EXEC,
))
_RUN_PROGRAM_BULLET_INTERACTIVE = """- Run preset hidden test cases (dataset IO, not your own cases):
                           <execute_bash>python solution.py</execute_bash>
"""
_ACTION_SPEC_INTERACTIVE = _with_examples(_ACTION_SPEC_COMMON.format(
    run_program_bullet=_RUN_PROGRAM_BULLET_INTERACTIVE,
    mode_rules=_RULES_INTERACTIVE,
))

_ACTION_SPEC = _ACTION_SPEC_BLIND  # default for imports / backward compat


def action_spec_for_mode(mode: str) -> str:
    """Return the ReAct action spec for ``test_feedback_mode``."""
    m = str(mode or "blind").lower()
    if m == "oracle":
        m = "interactive"
    if m == "exec":
        return _ACTION_SPEC_EXEC
    if m == "interactive":
        return _ACTION_SPEC_INTERACTIVE
    return _ACTION_SPEC_BLIND

SWEBENCH_TEMPLATE_NO_HIS = """
You are an expert software engineer solving a coding task with the ReAct pattern
(Thought → Action → Observation).

{repo_view}

{action_spec}
You are at step 1. Produce Thought, then Action, for this step only.
"""

SWEBENCH_TEMPLATE = """
You are an expert software engineer solving a coding task with the ReAct pattern
(Thought → Action → Observation).

## Task
{problem_statement}

## Prior ReAct trace (most recent {history_length} of {step_count} step(s))
{action_history}

## Current step {current_step}
{current_observation}

{action_spec}
Using the trace and current Observation, produce Thought, then Action, for this step only.
"""
