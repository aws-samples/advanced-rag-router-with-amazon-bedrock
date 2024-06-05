# Create a folder named 'python'
mkdir python

# cd into that folder
cd python

# Install the required dependencies
pip install opensearch-py -t ./
pip install requests -t ./
pip install requests-aws4auth -t ./

# cd out of that folder
cd ..

# Zip that folder
zip -r py312_opensearch-py_requests_and_requests-aws4auth.zip ./python