"""Resolve SBATCH defaults before rendering a job script."""

import shlex
from typing import Iterable, List


def resolve_sbatch_directives(*groups: Iterable[str], job_name: str = "") -> List[str]:
    """Preserve ordered directives and add defaults for omitted options.

    Each configuration entry contains one option and its optional argument.
    Both long-option forms and attached or separate short arguments are accepted.
    """
    directives = [directive for group in groups for directive in group]
    aliases = {"-J": "--job-name", "-o": "--output", "-e": "--error", "-A": "--account"}
    options = {}
    for directive in directives:
        words = shlex.split(directive)
        if not words:
            continue
        option, separator, value = words[0].partition("=")
        if option in aliases.values():
            if not separator:
                value = words[1] if len(words) > 1 else ""
        elif words[0][:2] in aliases:
            option = aliases[words[0][:2]]
            value = words[0][2:] or (words[1] if len(words) > 1 else "")
        else:
            continue
        options[option] = value

    defaults = []
    if job_name and "--job-name" not in options:
        defaults.append("--job-name=" + shlex.quote(job_name))
    account = options.get("--account", "")
    if account:
        for option, extension in (("--output", "out"), ("--error", "err")):
            if option not in options:
                defaults.append(option + "=" + shlex.quote(f"{account}-%x-%j.{extension}"))
    return defaults + directives
