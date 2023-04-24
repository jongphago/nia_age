FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

COPY	./requirements.txt /requirements.txt

RUN	apt-get update && apt-get install -y \
	python3-pip \
	libgl1-mesa-glx

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt
#
#EXPOSE	80 443
## 이 컨테이너가 해당 포트를 사용할 예정임을 사용자에게 알려준다.
## 실제로 포트를 열기 위해서는 run 명령어에서 -p 옵션을 사용해야한다.
#
#CMD 	bash run.sh
## 생성된 컨테이너를 실행할 명령어를 지정한다.