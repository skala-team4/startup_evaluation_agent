import os
import json
import re
from langchain.utils import TavilyClient
from openai import OpenAI
from state_definitions import InvestmentState  # 중앙 집중식 상태 모듈 임포트

# API 설정
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 클라이언트 초기화
tavily = TavilyClient(api_key=TAVILY_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

def market_research(state: InvestmentState) -> InvestmentState:
    """
    시장 연구 에이전트 - 스타트업 도메인의 시장 규모 및 잠재력을 분석합니다.
    """
    # 스타트업 정보 추출 (이전 에이전트에서 설정한 값)
    domain = state.get("startup_info", {}).get("domain", "헬스케어")
    startup_name = state.get("startup_info", {}).get("name", "")
    
    # 1. Tavily 웹 검색
    query = f"{domain} 시장 규모 성장률 트렌드 수익 모델 {startup_name}"
    try:
        result = tavily.search(query=query, search_depth="advanced")
        documents = result.get("results", [])[:3]  # 상위 3개 문서만 가져옴
        market_text = "\n\n".join([f"{doc['title']}\n{doc['content']}" for doc in documents])
    except Exception as e:
        print(f"시장 정보 검색 중 오류 발생: {str(e)}")
        market_text = f"시장 정보를 가져오지 못했습니다. 오류: {str(e)}"
        documents = []
    
    # 2. GPT 프롬프트 구성
    system_prompt = (
        "당신은 스타트업 투자 평가 전문가입니다. 아래 시장 관련 정보를 바탕으로 5가지 항목을 0~10점으로 평가하고 평가 이유를 함께 설명하세요.\n"
        f"대상 도메인: {domain}\n"
        f"스타트업: {startup_name}\n\n"
        "다음과 같은 JSON 형식으로 출력하세요:\n"
        "{\n"
        '  "market_size": {"score": int, "reasoning": string},\n'
        '  "problem_fit": {"score": int, "reasoning": string},\n'
        '  "willingness_to_pay": {"score": int, "reasoning": string},\n'
        '  "revenue_model_clarity": {"score": int, "reasoning": string},\n'
        '  "upside_potential": {"score": int, "reasoning": string},\n'
        '  "market_size_estimate": string,\n'
        '  "growth_rate_estimate": string,\n'
        '  "key_trends": [string],\n'
        '  "regulatory_concerns": [string]\n'
        "}"
    )
    
    user_prompt = f"시장 분석을 위한 정보:\n{market_text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",  # 최신 모델 사용
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        
        # 3. JSON 파싱
        raw_output = response.choices[0].message.content
        
        # JSON 형식 추출 (정규식 사용)
        json_match = re.search(r"\{[\s\S]*\}", raw_output)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            # JSON 형식이 아닌 경우 기본값 설정
            parsed = {
                "market_size": {"score": 5, "reasoning": "정보 부족으로 평균 점수 부여"},
                "problem_fit": {"score": 5, "reasoning": "정보 부족으로 평균 점수 부여"},
                "willingness_to_pay": {"score": 5, "reasoning": "정보 부족으로 평균 점수 부여"},
                "revenue_model_clarity": {"score": 5, "reasoning": "정보 부족으로 평균 점수 부여"},
                "upside_potential": {"score": 5, "reasoning": "정보 부족으로 평균 점수 부여"},
                "market_size_estimate": "정보 부족",
                "growth_rate_estimate": "정보 부족",
                "key_trends": ["정보 부족"],
                "regulatory_concerns": ["정보 부족"]
            }
            
    except Exception as e:
        print(f"시장 분석 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본값 설정
        parsed = {
            "market_size": {"score": 5, "reasoning": f"분석 중 오류: {str(e)}"},
            "problem_fit": {"score": 5, "reasoning": f"분석 중 오류: {str(e)}"},
            "willingness_to_pay": {"score": 5, "reasoning": f"분석 중 오류: {str(e)}"},
            "revenue_model_clarity": {"score": 5, "reasoning": f"분석 중 오류: {str(e)}"},
            "upside_potential": {"score": 5, "reasoning": f"분석 중 오류: {str(e)}"},
            "market_size_estimate": "분석 중 오류",
            "growth_rate_estimate": "분석 중 오류",
            "key_trends": ["분석 중 오류"],
            "regulatory_concerns": ["분석 중 오류"]
        }
    
    # 4. 시장 분석 데이터 생성
    market_scores = {
        "market_size": parsed.get("market_size", {"score": 5, "reasoning": ""}),
        "problem_fit": parsed.get("problem_fit", {"score": 5, "reasoning": ""}),
        "willingness_to_pay": parsed.get("willingness_to_pay", {"score": 5, "reasoning": ""}),
        "revenue_model_clarity": parsed.get("revenue_model_clarity", {"score": 5, "reasoning": ""}),
        "upside_potential": parsed.get("upside_potential", {"score": 5, "reasoning": ""})
    }
    
    # 5. 참조 문서 저장
    market_documents = []
    for doc in documents:
        market_documents.append({
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "content": doc.get("content", "")
        })
    
    # 6. 시장 분석 상태 업데이트
    if "market_analysis" not in state:
        state["market_analysis"] = {}
    
    state["market_analysis"].update({
        "market_scores": market_scores,
        "market_documents": market_documents,
        "market_size_estimate": parsed.get("market_size_estimate", "정보 없음"),
        "growth_rate_estimate": parsed.get("growth_rate_estimate", "정보 없음"),
        "key_trends": parsed.get("key_trends", []),
        "regulatory_concerns": parsed.get("regulatory_concerns", [])
    })
    
    # 평균 시장 점수 계산 (투자 결정에 활용)
    scores_only = [item.get("score", 5) for item in market_scores.values()]
    state["market_analysis"]["average_market_score"] = sum(scores_only) / len(scores_only)
    
    # 7. 상태 업데이트
    state["status"] = "market_research_completed"
    
    return state

# 랭그래프 노드 생성 함수 - 메인에서 임포트할 때 사용
def create_market_research_agent():
    from langgraph.graph import StateGraph
    
    graph = StateGraph(InvestmentState)
    graph.add_node("market_research", market_research)
    
    # 시작점과 종료점이 같은 단일 노드 그래프
    graph.set_entry_point("market_research")
    graph.set_finish_point("market_research")
    
    return graph.compile()

# 단독 실행 테스트용
if __name__ == "__main__":
    # 테스트용 초기 상태 생성
    test_state = {
        "startup_info": {
            "name": "MediHealth AI",
            "domain": "의료 AI 진단"
        },
        "competitors": [],
        "market_analysis": {},
        "investment_recommendation": {},
        "report_data": {},
        "status": "starting"
    }
    
    # 에이전트 실행
    result_state = market_research(test_state)
    
    # 결과 출력
    print(f"시장 규모 점수: {result_state['market_analysis']['market_scores']['market_size']['score']}")
    print(f"시장 규모 평가 이유: {result_state['market_analysis']['market_scores']['market_size']['reasoning']}")
    print(f"평균 시장 점수: {result_state['market_analysis']['average_market_score']}")
    print(f"시장 규모 추정치: {result_state['market_analysis']['market_size_estimate']}")
    print(f"주요 트렌드: {result_state['market_analysis']['key_trends']}")