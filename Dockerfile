FROM python:3.9

WORKDIR /StSciKitApp/Ex1

COPY requirements.txt .
COPY ./Src ./Src

RUN pip install -r requirements.txt

CMD ["python", "./Src/streamlit_ex.py"]