Project setup:
1. Create a virtualenv:
* pip3 install virtualenv
* virtualenv -p /usr/bin/python3 stepenv
* source stepenv/bin/activate
* pip3 install -r requirements.txt
* deactivate
2. Setup PyCharm:
New Project
* Location: /path/to/project/stepcount/src
* Interpreter: /path/to/project/stepcount/stepenv/bin/python3
* go to Preferences -> Project -> Project Structure
* + Add Content Root
* /path/to/project/stepcount/data


Working on project (in stepenv):
* source stepenv/bin/activate
* ... your work ...
* deactivate

Managing pip requirements
* pip3 freeze >requirements.txt # saving new requirements
* pip3 install -r requirements.txt # updating requirements
