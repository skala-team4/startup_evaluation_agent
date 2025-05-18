import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid
import time

# Pinecone API 키 설정
PINECONE_API_KEY = "pcsk_5eomHM_G5Qphjze7azah2CE7XnhygHsrVvp76ZzwtydLFFgFZtUR438teA7v2psLU16jFa"  # 실제 API 키의 일부만 표시됨

# CSV 파일 로드
csv_path = r"C:\Users\lucy8\skala\AI_project\data\startup_data.csv"
df = pd.read_csv(csv_path)

# 열 이름 변경
df = df.rename(columns={
    'startup': 'name',
    'text': 'summary'
})

print(f"CSV 파일에서 {len(df)}개의 레코드를 로드했습니다.")

# 임베딩 모델 로드
model_name = "all-MiniLM-L6-v2"  # 384차원 벡터 생성
model = SentenceTransformer(model_name)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"모델 임베딩 차원: {embedding_dim}")

# Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)

# 이미 생성된 인덱스에 연결
index_name = "startup-index"
index = pc.Index(index_name)
print(f"인덱스 '{index_name}'에 연결됨")

# 벡터화 및 업로드 함수
def upsert_to_pinecone(index, df, model):
    batch_size = 50  # 작은 배치 사이즈로 시작
    total_records = len(df)
    successful_uploads = 0
    
    for i in range(0, total_records, batch_size):
        end_idx = min(i + batch_size, total_records)
        # 배치 데이터 준비
        batch_df = df.iloc[i:end_idx]
        
        # 임베딩 생성
        summaries = batch_df['summary'].tolist()
        try:
            vectors = model.encode(summaries)
            
            # 업로드할 데이터 준비
            records = []
            for j, row in enumerate(batch_df.itertuples()):
                # ID 생성 (고유해야 함)
                id = str(uuid.uuid4())
                
                # 메타데이터 준비
                metadata = {
                    'name': str(row.name),  # 문자열로 변환
                    'summary': str(row.summary)  # 문자열로 변환
                }
                
                # 벡터 준비
                vector = vectors[j].tolist()
                
                # 레코드 추가
                records.append({
                    'id': id,
                    'values': vector,
                    'metadata': metadata
                })
            
            # Pinecone에 업로드
            index.upsert(vectors=records)
            
            successful_uploads += len(records)
            print(f"업로드 완료: {i+1}~{end_idx}/{total_records} 레코드 (총 성공: {successful_uploads})")
            # 속도 제한을 피하기 위한 짧은 대기
            time.sleep(1)
            
        except Exception as e:
            print(f"배치 {i}~{end_idx} 처리 중 오류 발생: {e}")
            time.sleep(5)  # 오류 후 더 긴 대기

# 메인 함수 실행
print(f"시작: {len(df)}개의 레코드를 Pinecone에 업로드합니다...")
upsert_to_pinecone(index, df, model)
print("완료: 모든 데이터가 Pinecone에 업로드되었습니다.")

# 인덱스 크기 확인
try:
    stats = index.describe_index_stats()
    print(f"업로드된 총 벡터 수: {stats['total_vector_count']}")
    print(f"인덱스 통계: {stats}")
except Exception as e:
    print(f"인덱스 통계 조회 중 오류 발생: {e}")