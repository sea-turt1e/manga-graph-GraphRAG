#!/usr/bin/env python3
"""
Main entry point for GraphRAG vs Standard LLM comparison

This module provides the main entry points for running comparisons between
GraphRAG and standard LLM approaches to manga recommendation and analysis.
"""

import os
import sys

from comparison_service import ComparisonFormatter, ComparisonService, interactive_comparison_demo
from demo_scenarios import DemoRunner, DemoScenarios


def main():
    """Main function to run comparison demos"""
    print("=== GraphRAG vs Standard LLM Comparison Tool ===\n")

    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--interactive":
            interactive_comparison_demo()
            return
        elif command == "--demo":
            run_predefined_demos()
            return
        elif command == "--custom":
            run_custom_demo()
            return
        elif command == "--help":
            print_help()
            return
        else:
            print(f"Unknown command: {command}")
            print_help()
            return

    # Default behavior: run predefined demos
    run_predefined_demos()


def run_predefined_demos():
    """Run predefined demo scenarios"""
    try:
        comparison_service = ComparisonService(api_base_url=os.getenv("API_BASE", "http://localhost:8000"))
        formatter = ComparisonFormatter()
        demo_runner = DemoRunner(comparison_service, formatter)

        print("実行中: 定義済みデモシナリオ")
        results = demo_runner.run_full_demo_suite()

        # 最終サマリーを表示
        summary = comparison_service.get_performance_summary()
        formatted_summary = formatter.format_performance_summary(summary)
        print("\n" + "=" * 50)
        print(formatted_summary)

        return results

    except Exception as e:
        print(f"❌ Error in predefined demos: {e}")
        print("\n対処方法:")
        print("1. Manga Graph APIが起動していることを確認してください")
        print("2. OpenAI APIキーが正しく設定されていることを確認してください")
        print("3. 必要なパッケージがインストールされていることを確認してください")


def run_custom_demo():
    """Run custom demo with user input"""
    print("=== カスタムデモモード ===")
    print("推薦クエリと分析タイトルを入力してください。")
    print("空行で入力を終了します。\n")

    # 推薦クエリの入力
    print("推薦クエリを入力してください:")
    recommendation_queries = []
    while True:
        query = input("Query (空行で終了): ").strip()
        if not query:
            break
        recommendation_queries.append(query)

    # 分析タイトルの入力
    print("\n分析タイトルを入力してください:")
    analysis_titles = []
    while True:
        title = input("Title (空行で終了): ").strip()
        if not title:
            break
        analysis_titles.append(title)

    if not recommendation_queries and not analysis_titles:
        print("入力がありません。定義済みデモを実行します。")
        run_predefined_demos()
        return

    try:
        comparison_service = ComparisonService(api_base_url=os.getenv("API_BASE", "http://localhost:8000"))
        formatter = ComparisonFormatter()
        demo_runner = DemoRunner(comparison_service, formatter)

        results = {}

        if recommendation_queries:
            print(f"\n推薦クエリ {len(recommendation_queries)} 件を実行中...")
            custom_scenarios = [{"query": query, "category": "カスタム"} for query in recommendation_queries]
            results["custom_recommendations"] = demo_runner.run_recommendation_demos(custom_scenarios)

        if analysis_titles:
            print(f"\n分析タイトル {len(analysis_titles)} 件を実行中...")
            custom_titles = [{"title": title, "category": "カスタム"} for title in analysis_titles]
            results["custom_analyses"] = demo_runner.run_analysis_demos(custom_titles)

        # サマリーを表示
        summary = comparison_service.get_performance_summary()
        formatted_summary = formatter.format_performance_summary(summary)
        print("\n" + "=" * 50)
        print(formatted_summary)

        return results

    except Exception as e:
        print(f"❌ Error in custom demo: {e}")


def run_specific_scenario(scenario_type: str):
    """Run specific type of scenario"""
    try:
        comparison_service = ComparisonService(api_base_url=os.getenv("API_BASE", "http://localhost:8000"))
        formatter = ComparisonFormatter()
        demo_runner = DemoRunner(comparison_service, formatter)

        if scenario_type == "recommendations":
            scenarios = DemoScenarios.get_recommendation_queries()
            results = demo_runner.run_recommendation_demos(scenarios)
        elif scenario_type == "analyses":
            titles = DemoScenarios.get_analysis_titles()
            results = demo_runner.run_analysis_demos(titles)
        elif scenario_type == "edge_cases":
            scenarios = DemoScenarios.get_edge_case_queries()
            results = demo_runner.run_edge_case_demos(scenarios)
        else:
            print(f"Unknown scenario type: {scenario_type}")
            return None

        return results

    except Exception as e:
        print(f"❌ Error in {scenario_type} scenarios: {e}")
        return None


def print_help():
    """Print help information"""
    print("使用方法:")
    print("  python main_comparison.py [オプション]")
    print()
    print("オプション:")
    print("  --demo        定義済みデモシナリオを実行（デフォルト）")
    print("  --interactive インタラクティブモードで実行")
    print("  --custom      カスタムクエリを入力して実行")
    print("  --help        このヘルプを表示")
    print()
    print("例:")
    print("  python main_comparison.py")
    print("  python main_comparison.py --interactive")
    print("  python main_comparison.py --custom")


def quick_test():
    """Quick test function for development"""
    print("=== クイックテスト ===")

    try:
        comparison_service = ComparisonService(api_base_url=os.getenv("API_BASE", "http://localhost:8000"))
        formatter = ComparisonFormatter()

        # 簡単なテスト
        test_query = "ONE PIECEが好きです。似たような作品を教えてください。"
        print(f"テストクエリ: {test_query}")

        result = comparison_service.compare_recommendations(test_query)
        formatted_output = formatter.format_comparison_result(result)
        print("\n" + formatted_output)

        return result

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return None


if __name__ == "__main__":
    # 開発時のクイックテスト用
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()
