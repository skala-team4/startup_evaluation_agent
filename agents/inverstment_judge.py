import os
import json
import re
from typing import Dict, Any
from openai import OpenAI
from state_definitions import InvestmentState

# OpenAI 클라이언트 초기화
client = None

def init_openai_client(api_key: str = None):
    """OpenAI 클라이언트 초기화"""
    global client
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다")
    client = OpenAI(api_key=api_key)
    return client

def investment_judgment(state: InvestmentState) -> InvestmentState:
    """
    투자 가능성 판단 에이전트 - 수집된 데이터를 바탕으로 투자 여부를 결정합니다.
    """
    global client
    if client is None:
        init_openai_client()
    
    # 예외 처리: 필요한 평가 정보 확인
    market_analysis = state.get("market_analysis", {})
    competitor_data = state.get("competitors", [])
    
    if not market_analysis:
        # 필요한 정보가 없는 경우 기본값 설정
        state["investment_recommendation"] = {
            "judgement": "불통과",
            "reasoning": "투자 판단에 필요한 시장 분석 데이터가 부족합니다."
        }
        state["status"] = "investment_judgment_insufficient_data"
        return state
    
    try:
        # 시장 점수 데이터 준비
        market_scores = market_analysis.get("market_scores", {})
        market_size_score = market_scores.get("market_size", {}).get("score", 5)
        problem_fit_score = market_scores.get("problem_fit", {}).get("score", 5)
        willingness_to_pay_score = market_scores.get("willingness_to_pay", {}).get("score", 5)
        revenue_model_clarity_score = market_scores.get("revenue_model_clarity", {}).get("score", 5)
        upside_potential_score = market_scores.get("upside_potential", {}).get("score", 5)
        
        # 경쟁사 데이터 준비
        competitive_score = market_analysis.get("competitive_score", 5)
        competitive_reasoning = market_analysis.get("competitive_reasoning", "정보 없음")
        
        # 스타트업 정보
        startup_info = state.get("startup_info", {})
        startup_name = startup_info.get("name", "미확인 스타트업")
        startup_domain = startup_info.get("domain", "기술")
        startup_summary = startup_info.get("summary", "정보 없음")
        
        # 시스템 프롬프트
        system_prompt = (
            "당신은 스타트업 투자 심사역입니다.\n"
            "아래 정보를 보고 '통과' 또는 '불통과' 중 하나로 투자 판단을 내려주세요.\n"
            "그리고 그 이유도 간단히 설명해주세요.\n"
            "출력은 반드시 다음 JSON 형식으로 해주세요:\n"
            "{\n"
            ' "judgement": "통과" 또는 "불통과",\n'
            ' "reasoning": "이유",\n'
            ' "score": 0부터 100까지의 투자 적합성 점수\n'
            "}"
        )
        
        # 사용자 프롬프트
        user_prompt = f"""
스타트업: {startup_name}
도메인: {startup_domain}
설명: {startup_summary}

[시장성 점수 (0-10점)]
- 시장 크기: {market_size_score}
- 문제 적합성: {problem_fit_score}
- 지불 의사: {willingness_to_pay_score}
- 수익 모델 명확성: {revenue_model_clarity_score}
- 성장 가능성: {upside_potential_score}

[경쟁사 분석]
- 경쟁력 점수 (0-10점): {competitive_score}
- 요약: {competitive_reasoning}

[경쟁사 정보]
"""
        # 경쟁사 정보 추가
        for i, competitor in enumerate(competitor_data[:3]):  # 상위 3개만
            user_prompt += f"""
경쟁사 {i+1}: {competitor.get('name', '미확인')}
강점: {', '.join(competitor.get('strengths', ['정보 없음']))}
약점: {', '.join(competitor.get('weaknesses', ['정보 없음']))}
"""
        
        # GPT 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        
        # GPT 응답 파싱
        raw_output = response.choices[0].message.content
        
        # JSON 파싱
        json_match = re.search(r"\{[\s\S]*\}", raw_output)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            # JSON 형식이 아닌 경우 기본값 설정
            parsed = {
                "judgement": "불통과",
                "reasoning": "응답 형식 오류: " + raw_output[:100] + "...",
                "score": 50
            }
        
        # 판단 결과 저장
        investment_recommendation = {
            "judgement": parsed.get("judgement", "불통과"),
            "reasoning": parsed.get("reasoning", "이유 없음"),
            "score": parsed.get("score", 50)
        }
        
        # 상태 업데이트
        state["investment_recommendation"] = investment_recommendation
        state["status"] = "investment_judgment_completed"
        
    except Exception as e:
        # 오류 처리
        state["investment_recommendation"] = {
            "judgement": "불통과",
            "reasoning": f"평가 중 오류 발생: {str(e)}",
            "score": 0
        }
        state["status"] = "investment_judgment_error"
    
    return state

# 랭그래프 노드 생성 함수 - 메인에서 임포트할 때 사용
def create_investment_judgment_agent():
    from langgraph.graph import StateGraph
    
    graph = StateGraph(InvestmentState)
    graph.add_node("investment_judgment", investment_judgment)
    
    # 시작점과 종료점이 같은 단일 노드 그래프
    graph.set_entry_point("investment_judgment")
    graph.set_finish_point("investment_judgment")
    
    return graph.compile()

# 단독 실행 테스트용
if __name__ == "__main__":
    # 초기화
    init_openai_client()
    
    # 테스트용 초기 상태
    test_state = {
        "startup_info": {
            "name": "MediHealth AI",
            "domain": "헬스케어",
            "summary": "AI를 활용한 원격 의료 진단 솔루션"
        },
        "market_analysis": {
            "market_scores": {
                "market_size": {"score": 8, "reasoning": "헬스케어 시장은 지속적으로 성장 중"},
                "problem_fit": {"score": 7, "reasoning": "원격 진료 수요가 높음"},
                "willingness_to_pay": {"score": 6, "reasoning": "보험사와 의료기관의 지불 의사 확인"},
                "revenue_model_clarity": {"score": 7, "reasoning": "구독 모델이 명확함"},
                "upside_potential": {"score": 9, "reasoning": "글로벌 확장 가능성이 높음"}
            },
            "competitive_score": 7,
            "competitive_reasoning": "기존 원격 진료 솔루션 대비 AI 정확도가 높음"
        },
        "competitors": [
            {
                "name": "TeleMed Solutions",
                "strengths": ["시장 점유율 높음", "브랜드 인지도"],
                "weaknesses": ["AI 기술 부족", "진단 정확도 낮음"]
            },
            {
                "name": "HealthAI",
                "strengths": ["선진 AI 기술", "대규모 투자 유치"],
                "weaknesses": ["제품 출시 지연", "비용 구조 높음"]
            }
        ],
        "status": "market_research_completed"
    }
    
    # 에이전트 실행
    result_state = investment_judgment(test_state)
    
    # 결과 출력
    print("\n=== 투자 판단 결과 ===")
    print(f"판단: {result_state['investment_recommendation']['judgement']}")
    print(f"이유: {result_state['investment_recommendation']['reasoning']}")
    print(f"점수: {result_state['investment_recommendation']['score']}")