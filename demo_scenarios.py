#!/usr/bin/env python3
"""
Demo scenarios for GraphRAG vs Standard LLM comparison

This module contains predefined demo scenarios and test cases for comparing
GraphRAG and standard LLM approaches to manga recommendation and analysis.
"""

import os
from typing import Any, Dict, List


class DemoScenarios:
    """Predefined demo scenarios for testing"""

    @staticmethod
    def get_recommendation_queries() -> List[Dict[str, Any]]:
        """Get predefined recommendation test queries"""
        return [
            {
                "id": "adventure_shounen",
                "query": "ONE PIECEが好きです。似たような冒険漫画を教えてください。",
                "category": "冒険・少年漫画",
                "expected_themes": ["冒険", "友情", "成長", "バトル"],
            },
            {
                "id": "ninja_action",
                "query": "NARUTOが好きです。忍者や友情をテーマにした作品を探しています。",
                "category": "忍者・アクション",
                "expected_themes": ["忍者", "友情", "成長", "努力"],
            },
            {
                "id": "psychological_thriller",
                "query": "心理戦やデスゲームが好きです。おすすめの作品はありますか？",
                "category": "心理戦・サスペンス",
                "expected_themes": ["心理戦", "デスゲーム", "サスペンス", "頭脳戦"],
            },
            {
                "id": "romance_shoujo",
                "query": "学園恋愛もので、心温まるストーリーが好きです。",
                "category": "学園・恋愛",
                "expected_themes": ["恋愛", "学園", "青春", "感動"],
            },
            {
                "id": "dark_fantasy",
                "query": "進撃の巨人のようなダークファンタジーが読みたいです。",
                "category": "ダークファンタジー",
                "expected_themes": ["ダークファンタジー", "謎", "絶望", "戦争"],
            },
            {
                "id": "sports_growth",
                "query": "スポーツ漫画で主人公の成長が描かれている作品を探しています。",
                "category": "スポーツ",
                "expected_themes": ["スポーツ", "成長", "努力", "チームワーク"],
            },
        ]

    @staticmethod
    def get_analysis_titles() -> List[Dict[str, Any]]:
        """Get predefined analysis test titles"""
        return [
            {"title": "ONE PIECE", "category": "冒険・少年漫画", "author": "尾田栄一郎", "genre": "冒険、アクション"},
            {"title": "NARUTO", "category": "忍者・アクション", "author": "岸本斉史", "genre": "忍者、アクション"},
            {
                "title": "進撃の巨人",
                "category": "ダークファンタジー",
                "author": "諫山創",
                "genre": "ダークファンタジー、アクション",
            },
            {"title": "鬼滅の刃", "category": "和風アクション", "author": "吾峠呼世晴", "genre": "和風、アクション"},
            {"title": "ドラゴンボール", "category": "バトル漫画", "author": "鳥山明", "genre": "バトル、アクション"},
        ]

    @staticmethod
    def get_edge_case_queries() -> List[Dict[str, Any]]:
        """Get edge case queries for testing system robustness"""
        return [
            {
                "id": "very_specific",
                "query": "主人公が料理人で、異世界に転生して、魔法を使って料理をする漫画はありますか？",
                "category": "非常に具体的な条件",
            },
            {
                "id": "contradictory",
                "query": "子供向けだけど大人も楽しめる、簡単だけど複雑なストーリーの漫画を教えてください。",
                "category": "矛盾する条件",
            },
            {"id": "vague", "query": "なんか面白い漫画ない？", "category": "曖昧な質問"},
            {
                "id": "non_existent",
                "query": "宇宙で寿司を握る忍者の漫画を探しています。",
                "category": "存在しない可能性の高い条件",
            },
        ]


class DemoRunner:
    """Demo execution and result management"""

    def __init__(self, comparison_service, formatter):
        """Initialize demo runner"""
        self.comparison_service = comparison_service
        self.formatter = formatter
        self.results = []

    def run_recommendation_demos(self, scenarios: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run recommendation demo scenarios"""
        if scenarios is None:
            scenarios = DemoScenarios.get_recommendation_queries()

        results = []

        print("=== 推薦機能デモ実行 ===")
        print(f"実行シナリオ数: {len(scenarios)}")
        print()

        for i, scenario in enumerate(scenarios, 1):
            print(f"--- シナリオ {i}: {scenario.get('category', 'Unknown')} ---")
            print(f"クエリ: {scenario['query']}")

            try:
                comparison_result = self.comparison_service.compare_recommendations(scenario["query"])
                comparison_result["scenario_info"] = scenario

                formatted_output = self.formatter.format_comparison_result(comparison_result)
                print(formatted_output)

                results.append(comparison_result)

            except Exception as e:
                error_result = {"scenario_info": scenario, "error": str(e), "success": False}
                results.append(error_result)
                print(f"❌ エラー: {e}")

            print("\n" + "=" * 70 + "\n")

        return results

    def run_analysis_demos(self, titles: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run analysis demo scenarios"""
        if titles is None:
            titles = DemoScenarios.get_analysis_titles()

        results = []

        print("=== 分析機能デモ実行 ===")
        print(f"実行タイトル数: {len(titles)}")
        print()

        for i, title_info in enumerate(titles, 1):
            print(f"--- 分析 {i}: {title_info.get('category', 'Unknown')} ---")
            print(f"作品: {title_info['title']}")

            try:
                comparison_result = self.comparison_service.compare_analyses(title_info["title"])
                comparison_result["title_info"] = title_info

                formatted_output = self.formatter.format_comparison_result(comparison_result)
                print(formatted_output)

                results.append(comparison_result)

            except Exception as e:
                error_result = {"title_info": title_info, "error": str(e), "success": False}
                results.append(error_result)
                print(f"❌ エラー: {e}")

            print("\n" + "=" * 70 + "\n")

        return results

    def run_edge_case_demos(self, scenarios: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run edge case scenarios"""
        if scenarios is None:
            scenarios = DemoScenarios.get_edge_case_queries()

        results = []

        print("=== エッジケーステスト実行 ===")
        print(f"実行シナリオ数: {len(scenarios)}")
        print()

        for i, scenario in enumerate(scenarios, 1):
            print(f"--- エッジケース {i}: {scenario.get('category', 'Unknown')} ---")
            print(f"クエリ: {scenario['query']}")

            try:
                comparison_result = self.comparison_service.compare_recommendations(scenario["query"])
                comparison_result["scenario_info"] = scenario
                comparison_result["is_edge_case"] = True

                formatted_output = self.formatter.format_comparison_result(comparison_result)
                print(formatted_output)

                results.append(comparison_result)

            except Exception as e:
                error_result = {"scenario_info": scenario, "error": str(e), "success": False, "is_edge_case": True}
                results.append(error_result)
                print(f"❌ エラー: {e}")

            print("\n" + "=" * 70 + "\n")

        return results

    def run_full_demo_suite(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run the full demo suite"""
        print("=== 完全デモスイート実行 ===")
        print("推薦、分析、エッジケースの全テストを実行します。")
        print()

        full_results = {
            "recommendations": self.run_recommendation_demos(),
            "analyses": self.run_analysis_demos(),
            "edge_cases": self.run_edge_case_demos(),
        }

        self._print_summary(full_results)

        return full_results

    def _print_summary(self, results: Dict[str, List[Dict[str, Any]]]):
        """Print summary of demo results"""
        print("=== デモ実行サマリー ===")

        for category, category_results in results.items():
            successful = sum(
                1 for r in category_results if r.get("comparison_metrics", {}).get("both_successful", False)
            )
            total = len(category_results)

            print(f"{category}: {successful}/{total} 成功")

            if successful < total:
                failed = [
                    r for r in category_results if not r.get("comparison_metrics", {}).get("both_successful", False)
                ]
                print(f"  失敗したケース: {len(failed)}")

        print()


def demo_comparison():
    """Main demo function - can be customized for different scenarios"""
    print("=== GraphRAG vs Standard LLM Comparison Demo ===\n")

    try:
        # これらのimportは実際の使用時にはメインファイルで行う
        from comparison_service import ComparisonFormatter, ComparisonService

        comparison_service = ComparisonService(api_base_url=os.getenv("API_BASE", "http://localhost:8000"))
        formatter = ComparisonFormatter()
        demo_runner = DemoRunner(comparison_service, formatter)

        # デフォルトのデモシナリオを実行
        results = demo_runner.run_full_demo_suite()

        return results

    except Exception as e:
        print(f"❌ Error in comparison demo: {e}")
        print("\n対処方法:")
        print("1. Manga Graph APIが起動していることを確認してください")
        print("2. OpenAI APIキーが正しく設定されていることを確認してください")
        print("3. 必要なパッケージがインストールされていることを確認してください")
        return None


def custom_demo(recommendation_queries: List[str] = None, analysis_titles: List[str] = None):
    """Custom demo with user-specified queries and titles"""
    print("=== カスタムデモ実行 ===\n")

    try:
        from comparison_service import ComparisonFormatter, ComparisonService

        comparison_service = ComparisonService(api_base_url=os.getenv("API_BASE", "http://localhost:8000"))
        formatter = ComparisonFormatter()
        demo_runner = DemoRunner(comparison_service, formatter)

        results = {}

        if recommendation_queries:
            print("カスタム推薦クエリを実行中...")
            custom_scenarios = [{"query": query, "category": "カスタム"} for query in recommendation_queries]
            results["custom_recommendations"] = demo_runner.run_recommendation_demos(custom_scenarios)

        if analysis_titles:
            print("カスタム分析タイトルを実行中...")
            custom_titles = [{"title": title, "category": "カスタム"} for title in analysis_titles]
            results["custom_analyses"] = demo_runner.run_analysis_demos(custom_titles)

        return results

    except Exception as e:
        print(f"❌ Error in custom demo: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--custom":
            # カスタムクエリの例
            custom_queries = ["サッカー漫画でおすすめはありますか？", "料理漫画が読みたいです"]
            custom_titles = ["キャプテン翼", "美味しんぼ"]
            custom_demo(custom_queries, custom_titles)
        else:
            print("Usage: python demo_scenarios.py [--custom]")
    else:
        demo_comparison()
