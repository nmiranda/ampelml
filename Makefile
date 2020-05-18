build:
	docker image build -t ampelmltest:1.0 ampelml/
	docker image build -t ampelml-web:1.0 ampelml-web/
	docker network create ampelml-net

delete:
	docker container rm --force ampelml ampelml-mongo ampelml-web
	docker network rm ampelml-net

start:
	docker stop ampelml ampelml-mongo ampelml-web

stop:
	docker stop ampelml ampelml-mongo ampelml-web

run:
	docker run -d --name ampelml-mongo --network ampelml-net mongo:bionic
	docker container run -d --publish 4242:4242 --name ampelml --network ampelml-net ampelmltest:1.0
	docker container run -d --publish 8080:8080 --name ampelml-web --network ampelml-net ampelml-web:1.0
	