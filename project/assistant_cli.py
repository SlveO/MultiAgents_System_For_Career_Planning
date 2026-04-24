from __future__ import annotations

import argparse
import json

try:
    from .assistant_schemas import TaskRequest, UserConstraints
    from .orchestrator import CareerOrchestrator
except ImportError:
    from project.assistant_schemas import TaskRequest, UserConstraints
    from project.orchestrator import CareerOrchestrator


def main():
    parser = argparse.ArgumentParser(description='多模态职业规划助手 CLI')
    parser.add_argument('--session-id', default='default-session')
    parser.add_argument('--goal', required=True, help='用户目标')
    parser.add_argument('--text', default='', help='补充文本')
    parser.add_argument('--images', nargs='*', default=[])
    parser.add_argument('--docs', nargs='*', default=[])
    parser.add_argument('--audio', nargs='*', default=[])
    parser.add_argument('--city', default=None)
    parser.add_argument('--education', default=None)
    parser.add_argument('--time-budget', type=int, default=None)
    parser.add_argument('--financial-budget', type=int, default=None)
    parser.add_argument('--brain-model', default=None)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--debug-trace', action='store_true')
    args = parser.parse_args()

    orchestrator = CareerOrchestrator()
    req = TaskRequest(
        session_id=args.session_id,
        user_goal=args.goal,
        text_input=args.text,
        image_paths=args.images,
        document_paths=args.docs,
        audio_paths=args.audio,
        brain_model=args.brain_model,
        stream=args.stream,
        debug_trace=args.debug_trace,
        constraints=UserConstraints(
            city=args.city,
            education_level=args.education,
            time_budget_hours_per_week=args.time_budget,
            financial_budget_cny=args.financial_budget,
        ),
    )

    if args.stream:
        final_result = None
        for evt in orchestrator.run_stream(req):
            event = evt.get('event')
            data = evt.get('data', {})
            if event == 'stage_start':
                print(f'\n[stage_start] {data}')
            elif event == 'stage_progress':
                print(f'[stage_progress] {data}')
            elif event == 'token':
                print(data.get('token', ''), end='', flush=True)
            elif event == 'stage_end':
                print(f'\n[stage_end] {data}')
            elif event == 'final_result':
                final_result = data

        print('\n')
        if final_result is not None:
            print(json.dumps(final_result, ensure_ascii=False, indent=2))
    else:
        response = orchestrator.run(req)
        print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
