import os
import logging
import argparse
import datetime
from typing import Dict, Any, List
from langgraph.graph import StateGraph
from state_definitions import GraphState

# 에이전트들 임포트
from agents.startup_explorer import create_startup_exploration_agent, init_resources as init_startup_resources
from agents.competitor_analyzer import create_competitor_analysis_agent
from agents.market_researcher import create_market_research_agent
from agents.inverstment_judge import create_investment_judgment_agent, init_openai_client
from agents.pdf_generator import create_pdf_generation_agent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("investment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_environment():
    """필요한 환경 변수와 리소스를 초기화합니다."""
    # 환경 변수 확인
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    
    # API 키 가져오기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    # 각 에이전트 초기화
    init_openai_client(openai_api_key)
    init_startup_resources(
        pinecone_api_key=pinecone_api_key,
        openai_api_key=openai_api_key,
        index_name="startup-index"  # Pinecone 인덱스 이름
    )
    
    logger.info("환경 초기화 완료")

def create_workflow_graph():
    """전체 투자 분석 워크플로우 그래프를 생성합니다."""
    graph = StateGraph(GraphState)
    
    # 각 에이전트 노드 생성
    startup_explorer = create_startup_exploration_agent()
    competitor_analyzer = create_competitor_analysis_agent()
    market_researcher = create_market_research_agent()
    investment_judge = create_investment_judgment_agent()
    pdf_generator = create_pdf_generation_agent()
    
    # 노드 추가
    graph.add_node("startup_exploration", startup_explorer)
    graph.add_node("competitor_analysis", competitor_analyzer)
    graph.add_node("market_research", market_researcher)
    graph.add_node("investment_judgment", investment_judge)
    graph.add_node("pdf_generation", pdf_generator)
    
    # 엣지 추가 (실행 흐름 정의)
    graph.add_edge("startup_exploration", "competitor_analysis")
    graph.add_edge("startup_exploration", "market_research")
    # 경쟁사 분석과 시장 조사가 모두 완료된 후 투자 판단
    graph.add_edge("competitor_analysis", "investment_judgment")
    graph.add_edge("market_research", "investment_judgment")
    # 투자 판단 후 PDF 생성
    graph.add_edge("investment_judgment", "pdf_generation")
    
    # 시작 노드 설정
    graph.set_entry_point("startup_exploration")
    
    # 컴파일 및 반환
    return graph.compile()

def run_investment_analysis(user_query: str) -> Dict[str, Any]:
    """
    사용자 쿼리에 따라 투자 분석을 수행하고 결과를 반환합니다.
    
    Args:
        user_query: 투자 평가를 수행할 스타트업 관련 검색어
        
    Returns:
        분석 결과가 담긴 상태 딕셔너리
    """
    # 환경 초기화
    initialize_environment()
    
    # 워크플로우 그래프 생성
    workflow = create_workflow_graph()
    
    # 초기 상태 생성
    initial_state = GraphState(
        user_query=user_query,
        domain=None,
        candidates_documents=[],
        evaluation_summary=None,
        market_analysis=None,
        market_scores=None,
        investment_decision=None,
        final_report=None
    )
    
    # 타임스탬프 추가 (PDF 보고서에 사용)
    initial_state_dict = initial_state.dict()
    initial_state_dict["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    logger.info(f"투자 분석 시작: '{user_query}'")
    
    # 워크플로우 실행
    try:
        result = workflow.invoke(initial_state_dict)
        logger.info("투자 분석 워크플로우 완료")
        return result
    except Exception as e:
        logger.error(f"워크플로우 실행 중 오류 발생: {str(e)}")
        raise

def print_analysis_result(result: Dict[str, Any]):
    """분석 결과를 콘솔에 출력합니다."""
    print("\n" + "="*50)
    print(" 스타트업 투자 분석 결과")
    print("="*50 + "\n")
    
    # 스타트업 정보
    startup_info = result.get("startup_info", {})
    print(f"▶ 스타트업: {startup_info.get('name', '정보 없음')}")
    print(f"▶ 도메인: {startup_info.get('domain', '정보 없음')}")
    print(f"▶ 설명: {startup_info.get('summary', '정보 없음')}")
    print("-"*50)
    
    # 시장 분석
    market_analysis = result.get("market_analysis", {})
    if market_analysis:
        print("\n▶ 시장 분석:")
        market_scores = market_analysis.get("market_scores", {})
        for key, value in market_scores.items():
            if isinstance(value, dict) and "score" in value:
                print(f"  • {key.replace('_', ' ').title()}: {value['score']}/10")
        
        print(f"\n▶ 경쟁력 점수: {market_analysis.get('competitive_score', '정보 없음')}/10")
    print("-"*50)
    
    # 투자 판단
    investment_recommendation = result.get("investment_recommendation", {})
    if investment_recommendation:
        print(f"\n▶ 투자 판단: {investment_recommendation.get('judgement', '정보 없음')}")
        print(f"▶ 점수: {investment_recommendation.get('score', '정보 없음')}/100")
        print(f"▶ 근거: {investment_recommendation.get('reasoning', '정보 없음')}")
    print("-"*50)
    
    # PDF 보고서
    report_data = result.get("report_data", {})
    if report_data:
        pdf_path = report_data.get("pdf_path", "생성 실패")
        print(f"\n▶ PDF 보고서: {pdf_path}")
    print("\n" + "="*50)

def main():
    """메인 프로그램 실행 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="AI 스타트업 투자 평가 시스템")
    parser.add_argument("--query", type=str, help="투자 평가를 수행할 스타트업 관련 검색어")
    args = parser.parse_args()
    
    # 사용자 쿼리 가져오기
    if args.query:
        user_query = args.query
    else:
        user_query = input("스타트업 검색어를 입력하세요 (예: AI 기반 헬스케어 스타트업): ")
    
    # 투자 분석 실행
    try:
        result = run_investment_analysis(user_query)
        print_analysis_result(result)
        
        # PDF 경로 확인
        report_data = result.get("report_data", {})
        pdf_path = report_data.get("pdf_path", "")
        
        if os.path.exists(pdf_path):
            print(f"\n✅ PDF 보고서가 성공적으로 생성되었습니다: {pdf_path}")
        else:
            print(f"\n❌ PDF 보고서 생성에 실패했습니다.")
            
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        print(f"\n❌ 오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)