# initialize conda environment
init:
	@echo "Setting up the virtual environment"
	conda env create -f environment.yml
	@echo 'Done!'

man_gen_env:
	@echo "Setting up the virtual environment"
	conda env create -f requirements.yml
	@echo 'Done!'

# create a directory
create_directory:
	@echo "Creating 'figs' folder, if it does not already exist."
	mkdir -p figs
	@echo "Done!"

# execute template
template:
	python -B src/template.py

# Generate Graphic 1
q1:
	@echo "Generating Graphic..." 	
	python -B src/q1.py
	@echo "Graphic generated!!"

# Generate Graphic 2
q2:
	@echo "Generating Graphic..." 	
	python -B src/q2.py
	@echo "Graphic generated!!"

# Generate Graphic 3
q3:
	@echo "Generating Graphic..." 	
	python -B src/q3.py
	@echo "Graphic generated!!"

# Generate Graphic 4
q4:
	@echo "Generating Graphic..." 	
	python -B src/q4.py
	@echo "Graphic generated!!"

# Execute GridSearch, generate graphic 5a
q5:
	@echo "Generating Graphic..." 	
	python -B src/q5.py
	@echo "Graphic generated!!"

# Execite model found in GridSearch, generate graphic 5b
q5b:
	@echo "Generating Graphic..." 	
	python -B src/q5b.py
	@echo "Graphic generated!!"

# Remove conda environment
clean:
	@echo "Removing env..."
	conda env remove -n assignment_env
	conda env remove -n quick_environment
	@echo "Done!!!"

.PHONY: init plot update
