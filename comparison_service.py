#!/usr/bin/env python3
"""
GraphRAG vs Non-GraphRAG Comparison Service

This module provides services to compare:
1. GraphRAG-based recommendations (using graph context)
2. Standard LLM-based recommendations (without graph context)
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from sample_with_manga_graph_api import MangaGraphRAG

# プロンプトを別ファイルからインポート
try:
    from prompts.manga_prompts import StandardMangaPrompts
except ImportError:
    print("Warning: prompts.manga_prompts not found. Using fallback prompts.")

    class StandardMangaPrompts:
        @staticmethod
        def get_recommendation_prompt():
            from langchain.prompts import PromptTemplate

            return PromptTemplate(
                input_variables=["user_query"], template="ユーザーの質問: {user_query}\n\n漫画を推薦してください。"
            )

        @staticmethod
        def get_analysis_prompt():
            from langchain.prompts import PromptTemplate

            return PromptTemplate(
                input_variables=["manga_title"], template="「{manga_title}」について分析してください。"
            )


load_dotenv()


class StandardMangaRecommender:
    """Standard manga recommender using only LLM without graph context"""

    def __init__(self):
        """Initialize the standard recommender"""
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
        self._init_prompts()

    def _init_prompts(self):
        """Initialize prompt templates for standard recommendations"""
        self.recommendation_prompt = StandardMangaPrompts.get_recommendation_prompt()
        self.analysis_prompt = StandardMangaPrompts.get_analysis_prompt()

    def recommend_manga(self, user_preference: str) -> Dict[str, Any]:
        """Generate manga recommendation using standard LLM approach"""
        recommendation_chain = self.recommendation_prompt | self.llm
        result = recommendation_chain.invoke({"user_query": user_preference})

        return {"recommendation": result.content, "method": "standard_llm", "context_used": "general_knowledge_only"}

    def analyze_manga(self, manga_title: str) -> Dict[str, Any]:
        """Analyze manga using standard LLM approach"""
        analysis_chain = self.analysis_prompt | self.llm
        result = analysis_chain.invoke({"manga_title": manga_title})

        return {"analysis": result.content, "method": "standard_llm", "context_used": "general_knowledge_only"}


class ComparisonService:
    """Service to compare GraphRAG and standard LLM approaches"""

    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """Initialize comparison service"""
        self.graphrag = MangaGraphRAG(api_base_url)
        self.standard = StandardMangaRecommender()
        self.metrics_history = []

    def compare_recommendations(self, user_query: str) -> Dict[str, Any]:
        """Compare GraphRAG and standard recommendations"""
        print(f"Comparing recommendations for: {user_query}")

        # Get GraphRAG recommendation
        graphrag_result = self._get_graphrag_recommendation(user_query)

        # Get standard LLM recommendation
        standard_result = self._get_standard_recommendation(user_query)

        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(graphrag_result, standard_result)

        # Store metrics for analysis
        self.metrics_history.append(
            {
                "type": "recommendation",
                "query": user_query,
                "metrics": comparison_metrics,
                "timestamp": self._get_timestamp(),
            }
        )

        return {
            "query": user_query,
            "graphrag_result": graphrag_result,
            "standard_result": standard_result,
            "comparison_metrics": comparison_metrics,
        }

    def compare_analyses(self, manga_title: str) -> Dict[str, Any]:
        """Compare GraphRAG and standard manga analysis"""
        print(f"Comparing analyses for: {manga_title}")

        # Get GraphRAG analysis
        graphrag_result = self._get_graphrag_analysis(manga_title)

        # Get standard analysis
        standard_result = self._get_standard_analysis(manga_title)

        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(graphrag_result, standard_result)

        # Store metrics for analysis
        self.metrics_history.append(
            {
                "type": "analysis",
                "title": manga_title,
                "metrics": comparison_metrics,
                "timestamp": self._get_timestamp(),
            }
        )

        return {
            "manga_title": manga_title,
            "graphrag_result": graphrag_result,
            "standard_result": standard_result,
            "comparison_metrics": comparison_metrics,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.metrics_history:
            return {"message": "No metrics available"}

        total_tests = len(self.metrics_history)
        successful_tests = sum(1 for m in self.metrics_history if m["metrics"].get("both_successful", False))

        graphrag_success_rate = (
            sum(1 for m in self.metrics_history if m["metrics"].get("graphrag_successful", False)) / total_tests
        )
        standard_success_rate = (
            sum(1 for m in self.metrics_history if m["metrics"].get("standard_successful", False)) / total_tests
        )

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "graphrag_success_rate": graphrag_success_rate,
            "standard_success_rate": standard_success_rate,
            "last_updated": self._get_timestamp(),
        }

    def _get_graphrag_recommendation(self, user_query: str) -> Dict[str, Any]:
        """Get recommendation using GraphRAG"""
        try:
            result = self.graphrag.recommend_from_text(user_query)
            return {
                "recommendation": result["recommendation"],
                "method": "graphrag",
                "context_used": "graph_database",
                "graph_data_size": len(result.get("graph_data", {}).get("nodes", [])),
                "entities_linked": len(result.get("linked_candidates", [])),
                "success": True,
                "raw_result": result,
            }
        except Exception as e:
            return {
                "recommendation": f"GraphRAG error: {str(e)}",
                "method": "graphrag",
                "context_used": "graph_database",
                "graph_data_size": 0,
                "entities_linked": 0,
                "success": False,
                "error": str(e),
            }

    def _get_standard_recommendation(self, user_query: str) -> Dict[str, Any]:
        """Get recommendation using standard LLM"""
        try:
            result = self.standard.recommend_manga(user_query)
            return {
                "recommendation": result["recommendation"],
                "method": "standard_llm",
                "context_used": "general_knowledge_only",
                "success": True,
                "raw_result": result,
            }
        except Exception as e:
            return {
                "recommendation": f"Standard LLM error: {str(e)}",
                "method": "standard_llm",
                "context_used": "general_knowledge_only",
                "success": False,
                "error": str(e),
            }

    def _get_graphrag_analysis(self, manga_title: str) -> Dict[str, Any]:
        """Get analysis using GraphRAG"""
        try:
            result = self.graphrag.analyze_manga(manga_title)
            return {
                "analysis": result["analysis"],
                "method": "graphrag",
                "context_used": "graph_database",
                "graph_data_size": len(result.get("graph_data", {}).get("nodes", [])),
                "success": True,
                "raw_result": result,
            }
        except Exception as e:
            return {
                "analysis": f"GraphRAG error: {str(e)}",
                "method": "graphrag",
                "context_used": "graph_database",
                "graph_data_size": 0,
                "success": False,
                "error": str(e),
            }

    def _get_standard_analysis(self, manga_title: str) -> Dict[str, Any]:
        """Get analysis using standard LLM"""
        try:
            result = self.standard.analyze_manga(manga_title)
            return {
                "analysis": result["analysis"],
                "method": "standard_llm",
                "context_used": "general_knowledge_only",
                "success": True,
                "raw_result": result,
            }
        except Exception as e:
            return {
                "analysis": f"Standard LLM error: {str(e)}",
                "method": "standard_llm",
                "context_used": "general_knowledge_only",
                "success": False,
                "error": str(e),
            }

    def _calculate_comparison_metrics(
        self, graphrag_result: Dict[str, Any], standard_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics for comparing the two approaches"""
        metrics = {
            "both_successful": graphrag_result.get("success", False) and standard_result.get("success", False),
            "graphrag_successful": graphrag_result.get("success", False),
            "standard_successful": standard_result.get("success", False),
        }

        if metrics["both_successful"]:
            # Calculate text length comparison
            graphrag_text = graphrag_result.get("recommendation", graphrag_result.get("analysis", ""))
            standard_text = standard_result.get("recommendation", standard_result.get("analysis", ""))

            metrics.update(
                {
                    "graphrag_length": len(graphrag_text),
                    "standard_length": len(standard_text),
                    "length_ratio": len(graphrag_text) / len(standard_text) if len(standard_text) > 0 else 0,
                    "graph_data_available": graphrag_result.get("graph_data_size", 0) > 0,
                    "entities_linked": graphrag_result.get("entities_linked", 0),
                }
            )

        return metrics

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime

        return datetime.now().isoformat()


class ComparisonFormatter:
    """Formatter for comparison results"""

    @staticmethod
    def format_comparison_result(comparison_result: Dict[str, Any]) -> str:
        """Format comparison result for display"""
        output = []

        # Header
        if "query" in comparison_result:
            output.append("=== 推薦比較結果 ===")
            output.append(f"ユーザークエリ: {comparison_result['query']}")
        elif "manga_title" in comparison_result:
            output.append("=== 分析比較結果 ===")
            output.append(f"分析対象: {comparison_result['manga_title']}")

        output.append("")

        # GraphRAG result
        output.append("【GraphRAG結果】")
        graphrag_result = comparison_result["graphrag_result"]
        if graphrag_result.get("success"):
            content = graphrag_result.get("recommendation", graphrag_result.get("analysis", ""))
            output.append(content)

            # Additional GraphRAG info
            if graphrag_result.get("graph_data_size", 0) > 0:
                output.append(f"\n[使用したグラフデータ: {graphrag_result['graph_data_size']}ノード]")
            if graphrag_result.get("entities_linked", 0) > 0:
                output.append(f"[リンクしたエンティティ: {graphrag_result['entities_linked']}個]")
        else:
            output.append(f"エラー: {graphrag_result.get('error', 'Unknown error')}")

        output.append("")
        output.append("=" * 50)
        output.append("")

        # Standard LLM result
        output.append("【標準LLM結果】")
        standard_result = comparison_result["standard_result"]
        if standard_result.get("success"):
            content = standard_result.get("recommendation", standard_result.get("analysis", ""))
            output.append(content)
        else:
            output.append(f"エラー: {standard_result.get('error', 'Unknown error')}")

        output.append("")
        output.append("=" * 50)
        output.append("")

        # Comparison metrics
        output.append("【比較メトリクス】")
        metrics = comparison_result["comparison_metrics"]

        if metrics.get("both_successful"):
            output.append(f"GraphRAG文字数: {metrics.get('graphrag_length', 0)}")
            output.append(f"標準LLM文字数: {metrics.get('standard_length', 0)}")
            output.append(f"文字数比率: {metrics.get('length_ratio', 0):.2f}")
            output.append(f"グラフデータ利用: {'あり' if metrics.get('graph_data_available') else 'なし'}")
            if metrics.get("entities_linked", 0) > 0:
                output.append(f"エンティティリンク数: {metrics['entities_linked']}")
        else:
            output.append(f"GraphRAG成功: {'○' if metrics.get('graphrag_successful') else '×'}")
            output.append(f"標準LLM成功: {'○' if metrics.get('standard_successful') else '×'}")

        return "\n".join(output)

    @staticmethod
    def format_performance_summary(summary: Dict[str, Any]) -> str:
        """Format performance summary for display"""
        if "message" in summary:
            return summary["message"]

        output = []
        output.append("=== パフォーマンスサマリー ===")
        output.append(f"総テスト数: {summary['total_tests']}")
        output.append(f"成功テスト数: {summary['successful_tests']}")
        output.append(f"成功率: {summary['success_rate']:.2%}")
        output.append(f"GraphRAG成功率: {summary['graphrag_success_rate']:.2%}")
        output.append(f"標準LLM成功率: {summary['standard_success_rate']:.2%}")
        output.append(f"最終更新: {summary['last_updated']}")

        return "\n".join(output)


def interactive_comparison_demo():
    """Interactive comparison demo"""
    print("=== Interactive GraphRAG vs Standard LLM Comparison ===")
    print("Available commands:")
    print("  compare-rec <query>  - Compare recommendations")
    print("  compare-ana <title>  - Compare analyses")
    print("  summary              - Show performance summary")
    print("  help                 - Show this help")
    print("  quit                 - Exit demo")
    print()

    try:
        comparison_service = ComparisonService(api_base_url=os.getenv("API_BASE", "http://localhost:8000"))
        formatter = ComparisonFormatter()

        while True:
            try:
                command = input("\nEnter command: ").strip()

                if command == "quit":
                    break
                elif command == "help":
                    print("Commands: compare-rec, compare-ana, summary, help, quit")
                elif command == "summary":
                    summary = comparison_service.get_performance_summary()
                    formatted_summary = formatter.format_performance_summary(summary)
                    print("\n" + formatted_summary)
                elif command.startswith("compare-rec "):
                    query = command[12:]
                    print(f"\nComparing recommendations for: {query}")
                    result = comparison_service.compare_recommendations(query)
                    formatted_output = formatter.format_comparison_result(result)
                    print("\n" + formatted_output)
                elif command.startswith("compare-ana "):
                    title = command[12:]
                    print(f"\nComparing analyses for: {title}")
                    result = comparison_service.compare_analyses(title)
                    formatted_output = formatter.format_comparison_result(result)
                    print("\n" + formatted_output)
                else:
                    print("Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\nThank you for using the comparison demo!")

    except Exception as e:
        print(f"❌ Failed to initialize comparison service: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_comparison_demo()
    else:
        # デモは別ファイルから実行
        try:
            from demo_scenarios import demo_comparison

            demo_comparison()
        except ImportError:
            print("Demo scenarios not found. Running interactive mode instead.")
            interactive_comparison_demo()
