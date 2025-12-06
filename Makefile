clean:
	rm -rf ./out
	rm -rf ./data

venv:
	python3 -m venv venv

format:
	python3 -m isort ./lorag
	python3 -m black ./lorag