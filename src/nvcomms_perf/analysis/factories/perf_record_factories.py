from typing import Optional, Any
import uuid
from ..models.metadata_sweep import (
    SweepMetadata,
    SystemMetadata,
    NetworkMetadata,
)
from ..models.metadata_test_set import (
    TestSetMetadata,
)
from ..models.metadata_test import (
    TestMetadata,
)
from ..models.nccl_tests import (
    NcclTestCmdOptions,
    NcclTestResult,
)
from ..models.perf_record import (
    PerfRecord,
    RecordSystemInfo,
    RecordNetworkInfo,
    RecordNcclTestDeviceInfo,
    RecordNcclTestsMetadataInfo,
    RecordNcclTestsConfig,
)


def _build_record_system_info(sysinfo: SystemMetadata) -> RecordSystemInfo:
    return RecordSystemInfo(
        system_name=sysinfo.system_name,
        gpu_arch_type=sysinfo.gpu_arch_type,
        cpu_model_name=sysinfo.cpu_model_name,
        cpu_arch_type=sysinfo.cpu_arch_type,
        os_type=sysinfo.os_type,
        os_version=sysinfo.os_version,
        linux_kernel_version=sysinfo.linux_kernel_version,
        adaptive_routing=sysinfo.adaptive_routing,
        cuda_driver_version=sysinfo.cuda_driver_version,
    )


def _build_record_network_info(netinfo: NetworkMetadata) -> RecordNetworkInfo:
    return RecordNetworkInfo(
        nics=netinfo.nics,
        switch_name=netinfo.switch_name,
        switch_type=netinfo.switch_type,
        switch_version=netinfo.switch_version,
        network_name=netinfo.network_name,
        network_version=netinfo.network_version,
        network_plugin_name=netinfo.network_plugin_name,
        network_plugin_version=netinfo.network_plugin_version,
        mofed_version=netinfo.mofed_version,
        libfabric_version=netinfo.libfabric_version,
    )


def _build_record_nccl_test_device_info(
    nccl_test_result: NcclTestResult,
) -> list[RecordNcclTestDeviceInfo]:
    return [
        RecordNcclTestDeviceInfo(
            rank=d.rank,
            group=d.group,
            pid=d.pid,
            hostname=d.hostname,
            device=d.device,
            device_hex=d.device_hex,
            device_info=d.device_info,
        )
        for d in nccl_test_result.config.devices
    ]


def _build_record_nccl_test_metadata_info(
    nccl_test_result: NcclTestResult, test_metadata: Optional[TestMetadata]
) -> RecordNcclTestsMetadataInfo:
    if test_metadata is None:
        return RecordNcclTestsMetadataInfo(
            start_time=nccl_test_result.start_time,
            end_time=nccl_test_result.end_time,
            devices=_build_record_nccl_test_device_info(nccl_test_result),
        )

    return RecordNcclTestsMetadataInfo(
        start_time=nccl_test_result.start_time or test_metadata.timestamp,
        end_time=nccl_test_result.end_time,
        devices=_build_record_nccl_test_device_info(nccl_test_result),
        default_config=test_metadata.default_config,
        iteration=test_metadata.iteration,
    )


def _build_record_nccl_test_env_vars(test_metadata: Optional[TestMetadata]) -> Optional[dict[str, str]]:
    if test_metadata is None or test_metadata.env is None:
        return None
    return {
        k.replace("xNCCL_", "NCCL_", 1) if (k.startswith("xNCCL_") and str(v).lower() == "default") else k:
        "default" if (k.startswith("xNCCL_") and str(v).lower() == "default") else v
        for k, v in test_metadata.env.items()
    }

def _build_record_nccl_test_config(
    nccl_test_result: NcclTestResult,
    test_metadata: Optional[TestMetadata] = None,
) -> RecordNcclTestsConfig:
    config_dict: dict[str, Any] = {}

    # If cmd_options exist, start with those values
    if nccl_test_result.cmd_options:
        config_dict.update(nccl_test_result.cmd_options.model_dump(exclude_none=True))
    elif test_metadata and test_metadata.cmd and "run_test" in test_metadata.cmd:
        # Try to parse from test_metadata.cmd if available
        _, _, cmdline = test_metadata.cmd.partition("run_test")
        cmdline = cmdline.strip()
        if cmdline:
            try:
                cmd_options = NcclTestCmdOptions.from_args(cmdline)
                cmd_dict = cmd_options.model_dump(exclude_none=True)
                config_dict.update(cmd_dict)
            except ValueError:
                pass

    # Always use values from nccl_test_result.config for required fields
    config = nccl_test_result.config
    config_dict["nthreads"] = config.n_threads
    config_dict["ngpus"] = config.n_gpus
    config_dict["minbytes"] = config.min_bytes
    config_dict["maxbytes"] = config.max_bytes
    config_dict["warmup_iters"] = config.warmup_iters
    config_dict["iters"] = config.iters
    config_dict["agg_iters"] = config.agg_iters
    config_dict["check"] = bool(config.validation) if config.validation is not None else False
    config_dict["cudagraph"] = config.graph

    # Optional fields
    if config.blocking_collectives is not None:
        config_dict["blocking"] = config.blocking_collectives
    if config.parallel_init is not None:
        config_dict["parallel_init"] = config.parallel_init
    # local_register comes from cmd_options if available

    return RecordNcclTestsConfig(**config_dict)


def build_perf_record_from_nccl_test_result(
    result: NcclTestResult,
    test_metadata: Optional[TestMetadata],
    test_set_metadata: Optional[TestSetMetadata],
    sweep_metadata: Optional[SweepMetadata],
    sweep_id: uuid.UUID,
    source: Optional[str],
    raw_data_path: Optional[str],
    hash: str,
) -> "PerfRecord":
    library_version = (
        result.nccl_version if result.nccl_version else test_set_metadata.library.version if test_set_metadata else None
    )
    return PerfRecord(
        # test properties
        name=test_metadata.test_name if test_metadata else None,
        description=None,
        log_file=test_metadata.log_name if test_metadata else None,
        # test set properties
        test_set_name=test_set_metadata.name if test_set_metadata else None,
        test_set_description=test_set_metadata.description if test_set_metadata else None,
        # sweep properties
        sweep_id=sweep_id,
        sweep_name=sweep_metadata.name if sweep_metadata else None,
        sweep_description=sweep_metadata.description if sweep_metadata else None,
        sweep_dir=sweep_metadata.raw_data_path if sweep_metadata else None,
        # test target
        tool="nccl-tests",
        tool_binary=test_metadata.test_binary if test_metadata else None,
        tool_version=None,
        library=test_set_metadata.library.name if test_set_metadata else "",
        library_binary=test_set_metadata.library.path if test_set_metadata else None,
        library_version=library_version,
        library_info={
            "commit_sha": test_set_metadata.library.commit_sha if test_set_metadata else None,
            "cuda_build_version": test_set_metadata.library.cuda_build_version if test_set_metadata else None,
            "cuda_runtime_version": test_set_metadata.library.cuda_runtime_version if test_set_metadata else None,
        },
        # test environment
        cluster_name=sweep_metadata.cluster_name if sweep_metadata else None,
        system_info=_build_record_system_info(sweep_metadata.system_metadata) if sweep_metadata else None,
        network_info=_build_record_network_info(sweep_metadata.network_metadata) if sweep_metadata else None,
        num_nodes=test_metadata.num_nodes if test_metadata else None,
        node_list=test_metadata.node_list if test_metadata else None,
        num_processes=test_metadata.num_processes if test_metadata else None,
        num_process_per_node=test_metadata.num_process_per_node if test_metadata else None,
        container=test_metadata.container if test_metadata else None,
        slurm_job_id=test_metadata.slurm_id if test_metadata else None,
        slurm_job_name=test_metadata.slurm_job_name if test_metadata else None,
        # test details
        test_cmd=test_metadata.cmd if test_metadata else None,
        test_metadata=_build_record_nccl_test_metadata_info(result, test_metadata),
        test_env_vars=_build_record_nccl_test_env_vars(test_metadata),
        test_config=_build_record_nccl_test_config(result, test_metadata),
        test_result=[row.flatten() for row in result.results],
        # miscellaneous
        source=source,
        collected_by=sweep_metadata.data_source if sweep_metadata else None,
        uploaded_by=sweep_metadata.data_reviewer if sweep_metadata else None,
        raw_data_path=raw_data_path,
        raw_data_hash=hash,
    )
