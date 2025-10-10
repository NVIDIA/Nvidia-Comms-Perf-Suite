..
   Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
   See LICENSE.txt for license information

Testing Automation
==================

Overview
--------

A framework for generating and running tests across different systems and configurations, supporting both baremetal and containerized environments.

Quick Start
-----------

1. Create your user configuration file (see :ref:`configuration-guide` below for advanced configuration):

   .. code-block:: toml

      CUDA_HOME = "/path/to/cuda"
      NCCL_HOME = "/path/to/nccl"
      NCCL_TEST_PATH = "/path/to/nccl_tests"

      # SBATCH directives for job submission
      sbatch = [
         "--gres=gpu:8",
         "--ntasks-per-node=8"
      ]

2. Create your testset configuration file, or use the pre-defined ones in ``testing-automation/config/nccl/``

   .. code-block:: toml

      # NCCL Test Configuration
      description = "allreduce_algo_proto_sweep"
      library_name = "nccl"

      [[test]]
      name = "all_reduce"
      binary = "all_reduce_perf"
      envs_merge_strategy = "override"
      args_merge_strategy = "override"
      [test.sweep_envs]
      NCCL_ALGO = ["RING", "TREE", "NVLSTREE"]
      NCCL_PROTO = ["SIMPLE", "LL128"]

3. Generate a job script:

   .. code-block:: bash

      # Generate a job script and print to stdout
      nvcomms-perf generate-job-script --system CLUSTER_1 --systems-toml systems.toml --user-toml user.toml --testset-toml testset.toml

      # Generate a job script and save to file
      nvcomms-perf generate-job-script --system CLUSTER_1 --systems-toml systems.toml --user-toml user.toml --testset-toml testset.toml --output job.sbatch

      # Generate a job script with specific run type and script type
      nvcomms-perf generate-job-script --system CLUSTER_1 --systems-toml systems.toml --user-toml user.toml --testset-toml testset.toml --run-type openmpi --script-type sbatch

4. Run the job script on cluster:

   .. code-block:: bash

      sbatch job.sbatch

Command-line Interface
----------------------

Arguments
~~~~~~~~~

- ``--system``: System name (e.g., CLUSTER_1, CLUSTER_2) **[required]**
- ``--systems-toml``: Path to systems TOML file **[required]**
- ``--user-toml``: Path to user TOML file **[required]**
- ``--testset-toml``: Path to testset TOML file **[required]**
- ``--run-type``: Run type (srun, container) [default: srun]
- ``--script-type``: Type of job script (sbatch, interactive, bash) [default: sbatch]
- ``--generate-test-report``: Generate test report after all tests are completed (multiple allowed, e.g., junit)
- ``--append-config-contents``: Append the contents of the TOML files to the generated job script
- ``--output``, ``-o``: Output path for job script. Default: Print to stdout
- ``--verbose``, ``-v``: Enable verbose output

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

- ``NVCOMMS_PERF_TOOLS_BINARY_SUFFIX``: Suffix to append to the binary name (e.g., "_mpi" for NCCL binaries in containerized environments). Suggested to include in user config. Can be set at run time.
- ``NVCOMMS_PERF_TOOLS_WORKSPACE``: Path to the workspace directory. Set at run time.

.. _configuration-guide:

Configuration Guide
-------------------

User Configuration
~~~~~~~~~~~~~~~~~~

Define your environment variables and settings in a TOML file, say ``user.toml``. The configuration supports four levels of precedence:

1. Library-specific (highest)
2. Running Environment-specific
3. System-specific
4. Global (lowest)

For most configuration values, the highest precedence level will override lower levels. However, for sbatch directives, the behavior is different:

- sbatch directives from different levels are merged in order of precedence
- This allows for combining sbatch directives from multiple levels
- For example, global sbatch directives can be combined with system-specific ones

Each config must be "name = value", they will be exported in the generated script

There are several special keywords in user configuration:

1. ``sbatch``: Each string in the list will be added as a SBATCH directive in the generated job script. SBATCH directives from different configuration levels are merged in order of precedence, allowing you to combine directives from multiple levels.

2. ``pyxis``: Each string in the list will be added as a Pyxis container parameter in the generated job script. This is only used when the run type is "container". Like ``sbatch``, Pyxis parameters from different levels are merged in order of precedence, allowing you to combine container parameters from multiple levels.

3. ``prologue_commands``: A string containing commands to be executed before the tests run. Unlike ``sbatch`` and ``pyxis``, this is not a list and follows the normal precedence rules where higher levels override lower levels.

4. ``epilogue_commands``: A string containing commands to be executed after all tests complete. Like ``prologue_commands``, this follows normal precedence rules where higher levels override lower levels.

Example user configuration with different precedence levels and sbatch/pyxis merge behavior:

.. code-block:: toml

   # Global configuration
   CUDA_HOME = "/path/to/your/cuda"
   LD_LIBRARY_PATH = "${CUDA_HOME}/lib64:${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:LD_LIBRARY_PATH}"

   # Global SBATCH directives
   sbatch = [
       "--job-name=nccl-test",
       "--nodes=2",
       "--ntasks-per-node=8"
   ]

   # Global Pyxis container parameters
   pyxis = [
       "--container-name=nccl-test",
       "--container-mounts=/path/to/data:/data"
   ]

   # Global prologue commands
   prologue_commands = """
   source /path/to/your/own/env/setup/script
   export PATH=${CUDA_HOME}/bin:${PATH} # you can also choose to export env vars here
   """

   # Global epilogue commands
   epilogue_commands = """
   echo "All tests completed"
   """

   [CLUSTER_1]
   CUDA_HOME = "/CLUSTER_1/path/to/your/cuda"

   # System-specific SBATCH directives
   sbatch = [
       "--partition=gpu",
       "--account=gpu"
   ]

   # System-specific Pyxis container parameters
   pyxis = [
       "--container-image=nccl-test:latest"
   ]

   # System-specific prologue commands
   prologue_commands = """
   echo "running tests on CLUSTER_1"
   """

   # System-specific epilogue commands
   epilogue_commands = """
   echo "tests completed on CLUSTER_1"
   """

   [CLUSTER_1.baremetal]
   [CLUSTER_1.baremetal.nccl]
   NCCL_HOME = "/CLUSTER_1/baremetal/path/to/your/nccl"
   NCCL_TEST_PATH = "/CLUSTER_1/baremetal/path/to/your/nccl-tests"
   MPI_HOME = "/CLUSTER_1/baremetal/path/to/your/openmpi"
   HPCX_HOME = "/CLUSTER_1/baremetal/path/to/your/hpcx"

   # Library-specific SBATCH directives
   sbatch = [
       "--gpus-per-node=8",
       "--time=1:00:00"
   ]

   # Library-specific Pyxis container parameters
   pyxis = [
       "--container-env=NCCL_DEBUG=WARN",
       "--container-env=NCCL_IB_TIMEOUT=20"
   ]

   [CLUSTER_1.container]
   NVCOMMS_PERF_TOOLS_BINARY_SUFFIX = "_mpi"

   # Library-specific prologue commands
   prologue_commands = """
   source $HPCX_HOME/hpcx-init.sh
   hpcx_load
   """

   # Library-specific epilogue commands
   epilogue_commands = """
   hpcx_unload
   """

Writing Testset Configuration
-----------------------------

The testset configuration file defines the tests to run and their parameters. Here's how to write your own configuration:

.. code-block:: toml

   # NCCL Test Configuration
   description = "allreduce_algo_proto_sweep_with_split_mask_user_buffer"
   library_name = "nccl"

   [common]
   path = "${NCCL_TEST_PATH}"

   [common.sweep_envs]
   NCCL_DEBUG = "WARN"
   NCCL_DEBUG_FILE = "nccl_debug_file.%h.%p.txt"
   NCCL_TESTS_SPLIT_MASK = ["0x0", "0x3", "0x7"]

   [common.sweep_args]
   -t = 1
   -g = 1
   -f = 2
   -n = 20
   -w = 5
   -b = "8"
   -e = "16G"
   -d = "float"
   -R = [0, 1]

   [common.sweep_custom_commands]
   pre_command = "export NCCL_MNNVL_CLIQUE_ID=$(($SLURM_PROCID / $CLIQUE_SIZE))"
   post_command = "unset NCCL_MNNVL_CLIQUE_ID"

   [[common.exceptions]]
   [common.exceptions.sweep_envs]
   NCCL_ALGO = ["NVLSTREE", "NVLS", "PAT"]
   NCCL_PROTO = ["LL", "LL128"]

   # Define test configurations for each collective
   [[test]]
   name = "all_reduce"
   binary = "all_reduce_perf"
   envs_merge_strategy = "override"
   args_merge_strategy = "override"
   pre_command = "export TEST_SPECIFIC_VAR=all_reduce_mode"
   post_command = "unset TEST_SPECIFIC_VAR"
   [test.sweep_envs]
   NCCL_ALGO = ["RING", "TREE", "NVLSTREE"]
   NCCL_PROTO = ["SIMPLE", "LL128"]

Key Components
~~~~~~~~~~~~~~

1. **Global Configuration**:
   - ``description``: Description of the testset
   - ``library_name``: Name of the library being tested (e.g., "nccl")
   - ``use_default_params``: Enable/disable default parameter validation
   - ``allowed_envs_to_sweep_on_default``: List of environment variables allowed to have multiple values, only takes effect on use_default_params = 1
   - ``allowed_args_to_sweep_on_default``: List of arguments allowed to have multiple values, only takes effect on use_default_params = 1
   - ``testset_iterations``: Number of times to repeat the entire testset (default: 1)
   - ``test_iterations``: Number of times to repeat each individual test (default: 1)

2. **Common Section**:
   - ``path``: Path to the test binaries (can use environment variables)
   - ``sweep_envs``: Environment variables applied to all tests
   - ``sweep_args``: Command line arguments applied to all tests
   - ``sweep_custom_commands``: Commands executed before/after each test across all tests
   - ``exceptions``: List of exceptions to filter out unwanted combinations

3. **Test Sections**:
   - Each test is defined under ``[[test]]``
   - ``name``: Name of the test
   - ``binary``: The executable to run
   - ``envs_merge_strategy``: How to merge environment variables with common section ("override" or "merge")
   - ``args_merge_strategy``: How to merge arguments with common section ("override" or "merge")
   - ``sweep_envs``: Test-specific environment variables
   - ``sweep_args``: Test-specific command line arguments
   - ``pre_command``: Command executed before each test runs (limited to maximum 1 value)
   - ``post_command``: Command executed after each test completes (limited to maximum 1 value)

4. **Parameter Sweeping**:
   - Parameters in ``sweep_envs`` and ``sweep_args`` can be single values or lists
   - Lists generate multiple combinations through Cartesian product
   - Example:

     .. code-block:: toml

        [test.sweep_envs]
        NCCL_ALGO = ["RING", "TREE"]    # Will generate 2 combinations
        NCCL_PROTO = ["SIMPLE", "LL128"] # Will generate 2 combinations
        # Total combinations: 2 * 2 = 4
        # 1. NCCL_ALGO=RING, NCCL_PROTO=SIMPLE
        # 2. NCCL_ALGO=RING, NCCL_PROTO=LL128
        # 3. NCCL_ALGO=TREE, NCCL_PROTO=SIMPLE
        # 4. NCCL_ALGO=TREE, NCCL_PROTO=LL128

        [test.sweep_args]
        -n = [2, 4]    # Will generate 2 combinations
        -g = [1, 2]    # Will generate 2 combinations
        # Total combinations: 2 * 2 = 4
        # 1. -n=2, -g=1
        # 2. -n=2, -g=2
        # 3. -n=4, -g=1
        # 4. -n=4, -g=2

   - Environment variables and arguments are swept independently
   - Total combinations = product of all list lengths in sweep_envs * product of all list lengths in sweep_args
   - Single values are used as-is in all combinations

5. **Pre and Post Commands**:
   - ``pre_command`` and ``post_command`` allow you to execute shell commands before and after each individual test
   - These are particularly useful for setting up test-specific environment variables or configurations
   - Commands are executed separately on each rank/process after spawning, enabling rank-specific setup
   - Each command is limited to a maximum of 1 value (no arrays allowed)
   - Can be defined in two locations:

     a. **Common Section** (applies to all tests):

     .. code-block:: toml

        [common.sweep_custom_commands]
        pre_command = "export NCCL_MNNVL_CLIQUE_ID=$(($SLURM_PROCID / $CLIQUE_SIZE))"
        post_command = "unset NCCL_MNNVL_CLIQUE_ID"

     b. **Test Section** (applies to specific test):

     .. code-block:: toml

        [[test]]
        name = "clique_test"
        binary = "all_reduce_perf"
        pre_command = "export TEST_SPECIFIC_VAR=value"
        post_command = "unset TEST_SPECIFIC_VAR"
        [test.sweep_envs]
        NCCL_ALGO = ["RING", "TREE"]

   - In these examples:
     - Before each test: Sets up environment variables based on test requirements
     - After each test: Cleans up the environment variables
     - Common pre/post commands apply to all tests, while test-specific commands apply only to that test

6. **Merge Strategies**:
   - ``override``: Test values override common values
   - ``merge``: Common and test values are combined into a list
   - Default is "override" if not specified

   Examples:

   .. code-block:: toml

      # Common section
      [common.sweep_envs]
      NCCL_ALGO = "RING"
      NCCL_PROTO = ["SIMPLE", "LL"]

      # Test with override strategy (default)
      [[test]]
      name = "test1"
      envs_merge_strategy = "override"
      [test.sweep_envs]
      NCCL_ALGO = "TREE"     # Overrides common value
      NCCL_PROTO = ["LL128"] # Overrides common value

      # Resulting environment variables:
      # NCCL_ALGO = "TREE"     (overridden)
      # NCCL_PROTO = ["LL128"] (overridden)

      # Test with merge strategy
      [[test]]
      name = "test2"
      envs_merge_strategy = "merge"
      [test.sweep_envs]
      NCCL_ALGO = "TREE"     # Merged with common value
      NCCL_PROTO = ["LL128"] # Merged with common value

      # Resulting environment variables:
      # NCCL_ALGO = ["RING", "TREE"]     (merged)
      # NCCL_PROTO = ["SIMPLE", "LL", "LL128"] (merged)

   The same applies to ``sweep_args`` with ``args_merge_strategy``:

   .. code-block:: toml

      # Common section
      [common.sweep_args]
      -n = 2
      -g = 1

      # Test with override strategy
      [[test]]
      name = "test3"
      args_merge_strategy = "override"
      [test.sweep_args]
      -n = 4    # Overrides common value
      -b = "8M" # New argument

      # Resulting arguments:
      # -n = 4     (overridden)
      # -g = 1     (from common)
      # -b = "8M"  (new)

      # Test with merge strategy
      [[test]]
      name = "test4"
      args_merge_strategy = "merge"
      [test.sweep_args]
      -n = 4    # Merged with common value
      -b = "8M" # New argument

      # Resulting arguments:
      # -n = [2, 4]  (merged)
      # -g = 1       (from common)
      # -b = "8M"    (new)

7. **Exceptions**:
   Exceptions allow you to filter out specific parameter combinations that you don't want to test.
   They can be defined in both common and test sections, and all exceptions are combined when
   processing.

   **Key Concepts**:

   - Exceptions are defined using the same structure as ``sweep_envs`` and ``sweep_args``
   - Multiple exceptions can be defined in both common and test sections
   - An empty list ``[]`` in an exception means "match any value"
   - If a parameter exists in both ``sweep_envs`` and ``sweep_args``, both must match to exclude a combination

   **Example Configuration**:

   .. code-block:: toml

      # Common exceptions - filter out specific algorithm/protocol combinations
      [[common.exceptions]]
      [common.exceptions.sweep_envs]
      NCCL_ALGO = ["NVLSTREE", "NVLS", "PAT"]  # Match any of these algorithms
      NCCL_PROTO = ["LL", "LL128"]             # Match any of these protocols

      # Test-specific exceptions - filter out specific mask values for PAT algorithm
      [[test.exceptions]]
      [test.exceptions.sweep_envs]
      NCCL_ALGO = "PAT"                        # Match PAT algorithm
      NCCL_TESTS_SPLIT_MASK = ["0x0", "0x3"]   # Match these mask values

   **Resulting Behavior**:

   The above configuration will exclude the following combinations:

   1. Any test where:
      - ``NCCL_ALGO`` is NVLSTREE, NVLS, or PAT **AND**
      - ``NCCL_PROTO`` is LL or LL128

   2. Any test where:
      - ``NCCL_ALGO`` is PAT **AND**
      - ``NCCL_TESTS_SPLIT_MASK`` is 0x0 or 0x3

   This effectively prevents testing certain algorithm/protocol combinations that are known
   to be incompatible or not interesting for your test cases.

Best Practices
~~~~~~~~~~~~~~

1. **Parameter Organization**:
   - Group related parameters together
   - Use descriptive names for tests
   - Keep common parameters in the common section
   - Use comments to explain parameter choices

2. **Sweeping Strategy**:
   - Be mindful of the number of combinations generated
   - Use arrays only when you need to test multiple values
   - Consider using exceptions to filter unwanted combinations
   - For default configurations, limit sweeping to NCCL_TESTS_SPLIT_MASK

3. **Environment Variables**:
   - Use ``${VARIABLE}`` syntax for paths and directories
   - Set appropriate debug levels (e.g., "WARN" for production)
   - Configure debug file paths for troubleshooting

4. **Pre and Post Commands**:
   - Use ``pre_command`` and ``post_command`` for test-specific setup and cleanup
   - Define common pre/post commands in ``[common.sweep_custom_commands]`` for actions needed by all tests
   - Define test-specific pre/post commands directly in test sections for test-unique requirements
   - Always pair setup commands with corresponding cleanup commands
   - Keep commands simple and focused on single operations
   - Use environment variable cleanup in post commands to avoid test interference

5. **Documentation**:
   - Add comments explaining parameter choices
   - Document any special requirements or dependencies
   - Note any known issues or limitations
   - Include examples of generated commands

Troubleshooting
---------------

1. **Missing Configuration Files**: Ensure all required configuration files are present in the ``config`` directory.
2. **Invalid System Name**: Check that the system name provided matches a system defined in ``systems.toml``.
3. **Invalid Run Type**: Ensure the run type is one of: ``srun`` or ``container``.
4. **Environment Variable Conflicts**: Check for conflicts in environment variables defined in different configuration files.
