clean:
	rm -rf ./out
	rm -rf ./data

venv:
	python3 -m venv venv

reqs:
	python3 -m pip install -r requirements.txt

format:
	python3 -m isort ./lorag
	python3 -m black ./lorag
	python3 -m pylint --recursive=y ./lorag