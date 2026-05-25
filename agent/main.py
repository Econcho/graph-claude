import argparse

from agent.config import build_default_model
from agent.runtime import AgentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal LangGraph coding agent.")
    parser.add_argument("--thread", required=True, help="LangGraph thread id.")
    parser.add_argument("--workspace", required=True, help="Workspace directory.")
    parser.add_argument("--input", required=True, help="User task.")
    parser.add_argument(
        "--permission-mode",
        default="default",
        choices=["default", "plan", "accept_edits"],
        help="Tool permission mode.",
    )
    parser.add_argument("--observe-dir", default=".runs", help="Runtime trace output directory.")
    parser.add_argument(
        "--observe-config",
        default=None,
        help="JSON config file that enables or disables observe hooks.",
    )
    parser.add_argument("--no-observe", action="store_true", help="Disable runtime trace output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = AgentRunner(
        llm_client=build_default_model(),
        observe=not args.no_observe,
        observe_dir=args.observe_dir,
        observe_config=args.observe_config,
    )
    result = runner.run(
        {
            "user_input": args.input,
            "workspace_root": args.workspace,
            "permission_mode": args.permission_mode,
            "session_id": args.thread,
            "agent_id": None,
            "auto_memory_enabled": True,
            "final_answer": None,
        },
        config={"configurable": {"thread_id": args.thread}},
    )
    print(result.get("final_answer") or "")


if __name__ == "__main__":
    main()
