import model_util as mu
import predictor

model = mu.getModel('test5')

first = '광지원초는 2005년부터 중국어 특성화학교로 지정돼 학생과 학부모 대상의 중국어 교육을 한다. 추격매수할 요인이 없어 추가강세 또한 조심스러울 수밖에'
second = '없다고 진단했다'

predictor.predict(first, second, model)