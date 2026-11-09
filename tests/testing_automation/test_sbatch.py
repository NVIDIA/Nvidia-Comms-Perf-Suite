"""SBATCH defaults must preserve explicit configuration and valid option forms."""

import os
import shlex
import subprocess

import pytest
import typer
from typer.testing import CliRunner

from nvcomms_perf.testing_automation.cli import generate_job_script_cmd
from nvcomms_perf.testing_automation.sbatch import resolve_sbatch_directives
from nvcomms_perf.testing_automation.script_generator import RunType, ScriptType, generate_job_script

SYSTEMS = '[cluster.run_params]\nsbatch = ["--account project"]\n'
USER = 'TEST_ENV = "value"\n'
TESTSET = """
library_name = "nccl"
sbatch = ["--exclusive"]
[[test]]
name = "smoke"
binary = "/bin/true"
sweep_envs = { TEST_VALUE = "1" }
sweep_args = { "-n" = "1" }
"""


@pytest.fixture
def script():
    return generate_job_script(SYSTEMS, USER, TESTSET, "cluster", RunType.SRUN, ScriptType.SBATCH, testset_name="sweep")


@pytest.mark.parametrize("account", ["--account=project", "--account project", "-Aproject", "-A project"])
def test_account_option_forms_produce_the_same_defaults(account):
    assert resolve_sbatch_directives([account], job_name="sweep") == [
        "--job-name=sweep",
        "--output=project-%x-%j.out",
        "--error=project-%x-%j.err",
        account,
    ]


@pytest.mark.parametrize(
    "explicit",
    [
        ["--job-name=chosen", "--output=chosen.out", "--error=chosen.err"],
        ["--job-name chosen", "--output chosen.out", "--error chosen.err"],
        ["-Jchosen", "-ochosen.out", "-echosen.err"],
        ["-J chosen", "-o chosen.out", "-e chosen.err"],
        ["  --job-name='chosen job'", "--output='chosen output'", "--error='chosen error'"],
    ],
)
def test_explicit_options_are_preserved(explicit):
    directives = ["--account=project", *explicit]
    assert resolve_sbatch_directives(directives, job_name="fallback") == directives


def test_last_account_across_configuration_layers_names_the_logs():
    system = ["--account system", "--exclusive"]
    user = ["-A user"]
    testset = ["--account=testset", "--open-mode=append"]
    assert resolve_sbatch_directives(system, user, testset, job_name="sweep") == [
        "--job-name=sweep",
        "--output=testset-%x-%j.out",
        "--error=testset-%x-%j.err",
        *system,
        *user,
        *testset,
    ]
    assert system == ["--account system", "--exclusive"]


def test_no_account_keeps_slurm_log_defaults():
    assert resolve_sbatch_directives(["--exclusive"], job_name="sweep") == [
        "--job-name=sweep",
        "--exclusive",
    ]
    assert resolve_sbatch_directives([]) == []


def test_default_job_name_with_spaces_is_one_argument():
    directive = resolve_sbatch_directives([], job_name="my sweep")[0]
    assert shlex.split(directive) == ["--job-name=my sweep"]


def test_generated_script_has_resolved_directives_before_shell_commands(script):
    assert script.startswith("#!/bin/bash\n")
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH")]
    assert directives == [
        "#SBATCH --job-name=sweep",
        "#SBATCH --output=project-%x-%j.out",
        "#SBATCH --error=project-%x-%j.err",
        "#SBATCH --account project",
        "#SBATCH --exclusive",
    ]
    assert script.index("#SBATCH --exclusive") < script.index("function validate_running_environment")
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr


def test_cli_uses_config_environment_defaults_and_testset_filename(tmp_path):
    systems = tmp_path / "systems.toml"
    user = tmp_path / "user.toml"
    testset = tmp_path / "my-sweep.toml"
    output = tmp_path / "job.sbatch"
    for path, content in ((systems, SYSTEMS), (user, USER), (testset, TESTSET)):
        path.write_text(content)
    prefix = generate_job_script.__module__.split(".")[0].upper()
    app = typer.Typer()
    app.command()(generate_job_script_cmd)
    result = CliRunner().invoke(
        app,
        ["--system", "cluster", "--testset-toml", str(testset), "--output", str(output)],
        env={prefix + "_SYSTEMS_TOML": str(systems), prefix + "_USER_TOML": str(user)},
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "#SBATCH --job-name=my-sweep\n" in output.read_text()
    assert "#SBATCH --exclusive\n" in output.read_text()


def test_test_failure_survives_logging_pipeline(script, tmp_path):
    function = script.split("function run_test()", 1)[1].split("export -f run_test", 1)[0]
    log_dir = tmp_path / "logs"
    info = log_dir / "nvl_domain_info"
    info.mkdir(parents=True)
    (info / "rank_1.txt").write_text("ClusterUUID: fixture\n")
    result = subprocess.run(
        ["bash"],
        input="function run_test()" + function + "\nrun_test bash -c 'exit 7'\n",
        text=True,
        capture_output=True,
        env={**os.environ, "LOG_DIR": str(log_dir), "STEP_NAME": "smoke", "TEST_NAME": "smoke", "SLURM_PROCID": "1"},
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert "Error: Test smoke failed" in result.stderr


@pytest.mark.parametrize("test_status,post_command,expected", [(7, "true", 7), (0, "(exit 9)", 9), (7, "(exit 9)", 7)])
def test_post_command_preserves_test_failure(test_status, post_command, expected):
    testset = TESTSET + '\n[test.sweep_custom_commands]\npost_command = "' + post_command + '"\n'
    rendered = generate_job_script(SYSTEMS, USER, testset, "cluster", RunType.SRUN, ScriptType.SBATCH)
    launch = next(
        line for line in rendered.splitlines() if line.lstrip().startswith("srun ") and "run_test_status=$?" in line
    )
    command = shlex.split(launch.split(" bash -c ", 1)[1])[0]
    result = subprocess.run(["bash", "-c", f"run_test() {{ return {test_status}; }}\n" + command], capture_output=True)
    assert result.returncode == expected, result.stderr


def test_validation_cleanup_preserves_other_results(script, tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    launcher = tools / "srun"
    launcher.write_text('#!/bin/bash\n[[ "$*" != *validate_running_environment* ]]\n')
    launcher.chmod(0o755)
    sweep = tmp_path / "validation_baremetal-nccl_tester_42"
    retained = sweep / "earlier-testset" / "evidence.txt"
    retained.parent.mkdir(parents=True)
    retained.write_text("keep this result")
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "USER": "tester",
            "SLURM_JOB_NAME": "validation",
            "SLURM_JOBID": "42",
            "SLURM_JOB_ID": "42",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1",
        },
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Error: Failed to validate running environment" in result.stderr
    assert retained.read_text() == "keep this result"
    assert list(sweep.iterdir()) == [retained.parent]
