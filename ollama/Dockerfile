FROM ollama/ollama:latest

ENV DEBIAN_FRONTEND=noninteractive

ENV OLLAMA_ORIGINS=*

RUN apt-get install -y curl nano wget

WORKDIR /app

COPY . .

RUN chmod u+x entrypoint.sh

EXPOSE 11434

ENTRYPOINT [ "bash", "entrypoint.sh" ]
