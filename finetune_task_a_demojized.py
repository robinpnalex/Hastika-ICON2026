#!/usr/bin/env python3
"""Fine-tune Task A with emoji converted to English descriptions."""

import sys

from finetune_task_a import main


def option_present(arguments, option):
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def demojized_arguments(arguments):
    arguments = list(arguments)
    if not option_present(arguments, "--output-dir"):
        arguments.extend(
            ["--output-dir", "checkpoints/muril_task_a_demojized"]
        )
    if not option_present(arguments, "--demojize"):
        arguments.append("--demojize")
    return arguments


if __name__ == "__main__":
    main(demojized_arguments(sys.argv[1:]))
