import os
import logging
from typing import Dict, Any
import markdown2
from weasyprint import HTML
from langgraph.graph import StateGraph

# state_definitions.py에서 정의한 상태 가져오기
from state_definitions import InvestmentState

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_markdown_from_state(state: Dict[str, Any]) -> str:
    """
    상태 정보를 기반으로 마크다운 보고서 콘텐츠를 생성합니다.
    """
    # 스타트업 정보 가져오기
    startup_info = state.get("startup_info", {})
    startup_name = startup_info.get("name", "미확인 스타트업")
    startup_domain = startup_info.get("domain", "기술")
    startup_summary = startup_info.get("summary", "정보 없음")
    
    # 시장 분석 정보 가져오기
    market_analysis = state.get("market_analysis", {})
    market_scores = market_analysis.get("market_scores", {})
    market_size_estimate = market_analysis.get("market_size_estimate", "정보 없음")
    growth_rate_estimate = market_analysis.get("growth_rate_estimate", "정보 없음")
    key_trends = market_analysis.get("key_trends", ["정보 없음"])
    
    # 경쟁사 정보 가져오기
    competitors = state.get("competitors", [])
    
    # 투자 판단 가져오기
    investment_recommendation = state.get("investment_recommendation", {})
    judgement = investment_recommendation.get("judgement", "판단 불가")
    reasoning = investment_recommendation.get("reasoning", "이유 없음")
    score = investment_recommendation.get("score", 0)
    
    # 마크다운 형식의 보고서 생성
    markdown_content = f"""
# 스타트업 투자 평가 보고서

## 1. 기본 정보

**스타트업명**: {startup_name}  
**도메인**: {startup_domain}  
**요약**: {startup_summary}

## 2. 시장 분석

**시장 규모**: {market_size_estimate}  
**성장률**: {growth_rate_estimate}

### 시장 점수 평가

| 평가 항목 | 점수 | 설명 |
|----------|------|------|
"""
    
    # 시장 점수 테이블 채우기
    for key, data in market_scores.items():
        if isinstance(data, dict):
            score = data.get("score", "-")
            reasoning = data.get("reasoning", "정보 없음")
            markdown_content += f"| {key.replace('_', ' ').title()} | {score} | {reasoning} |\n"
    
    # 주요 트렌드 추가
    markdown_content += "\n### 주요 트렌드\n\n"
    for trend in key_trends:
        markdown_content += f"* {trend}\n"
    
    # 경쟁사 분석
    markdown_content += "\n## 3. 경쟁사 분석\n\n"
    
    if competitors:
        for i, competitor in enumerate(competitors):
            comp_name = competitor.get("name", f"경쟁사 {i+1}")
            strengths = competitor.get("strengths", ["정보 없음"])
            weaknesses = competitor.get("weaknesses", ["정보 없음"])
            
            markdown_content += f"### {comp_name}\n\n"
            markdown_content += "**강점**:\n"
            for strength in strengths:
                markdown_content += f"* {strength}\n"
            
            markdown_content += "\n**약점**:\n"
            for weakness in weaknesses:
                markdown_content += f"* {weakness}\n"
            
            markdown_content += "\n"
    else:
        markdown_content += "경쟁사 정보가 없습니다.\n"
    
    # 투자 판단
    markdown_content += f"""
## 4. 투자 판단

**결정**: {judgement}  
**투자 적합성 점수**: {score}/100  
**판단 이유**: {reasoning}

---
본 보고서는 AI 기반 투자 평가 시스템에 의해 자동 생성되었습니다.  
생성일: {state.get("timestamp", "미정")}
"""
    
    return markdown_content

def pdf_generation(state: InvestmentState, output_path: str = "investment_report.pdf") -> InvestmentState:
    """
    상태 정보를 기반으로 투자 평가 PDF 보고서를 생성합니다.
    """
    try:
        # 1. Markdown 생성
        markdown_content = generate_markdown_from_state(state)
        
        # 2. Markdown → HTML로 변환
        html_content = markdown2.markdown(
            markdown_content,
            extras=["tables", "fenced-code-blocks"]
        )
        
        # HTML 스타일 추가
        styled_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #2980b9; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #ffffcc; padding: 2px 5px; }}
                .judgement {{ font-weight: bold; font-size: 18px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # 3. HTML → PDF로 변환
        try:
            HTML(string=styled_html).write_pdf(output_path)
            report_path = os.path.abspath(output_path)
            logger.info(f"PDF 보고서 생성 완료: {report_path}")
        except Exception as e:
            logger.error(f"PDF 생성 중 오류 발생: {str(e)}")
            report_path = "생성 실패"
        
        # 4. 상태에 저장 (PDF 경로)
        if "report_data" not in state:
            state["report_data"] = {}
        
        state["report_data"]["pdf_path"] = report_path
        state["report_data"]["markdown_content"] = markdown_content
        state["status"] = "pdf_generation_completed"
        
    except Exception as e:
        logger.error(f"보고서 생성 중 오류 발생: {str(e)}")
        
        if "report_data" not in state:
            state["report_data"] = {}
        
        state["report_data"]["error"] = str(e)
        state["status"] = "pdf_generation_error"
    
    return state

# 랭그래프 노드 생성 함수 - 메인에서 임포트할 때 사용
def create_pdf_generation_agent():
    from langgraph.graph import StateGraph
    
    graph = StateGraph(InvestmentState)
    
    # PDF 생성 함수에 파라미터 설정을 위한 래퍼 함수
    def pdf_gen_wrapper(state):
        return pdf_generation(state, "investment_report.pdf")
    
    graph.add_node("pdf_generation", pdf_gen_wrapper)
    
    # 시작점과 종료점이 같은 단일 노드 그래프
    graph.set_entry_point("pdf_generation")
    graph.set_finish_point("pdf_generation")
    
    return graph.compile()

# 단독 실행 테스트용
if __name__ == "__main__":
    import datetime
    
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
            "market_size_estimate": "2025년까지 약 3,000억 달러",
            "growth_rate_estimate": "연간 15-20%",
            "key_trends": [
                "AI 기반 진단 정확도 향상",
                "원격 의료 서비스 수요 증가",
                "의료 데이터 규제 강화"
            ]
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
        "investment_recommendation": {
            "judgement": "통과",
            "reasoning": "원격 의료 시장의 성장성과 AI 기술의 차별성을 고려할 때, 투자 가치가 있습니다.",
            "score": 78
        },
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "investment_judgment_completed"
    }
    
    # 에이전트 실행
    result_state = pdf_generation(test_state)
    
    # 결과 출력
    print("\n=== PDF 생성 결과 ===")
    print(f"상태: {result_state['status']}")
    print(f"PDF 경로: {result_state['report_data'].get('pdf_path', '생성 실패')}")