PATH=../:$PATH
mkdir -p ./notebooks
cp ../notebooks/*.ipynb ./notebooks
mkdir -p ./fig
cp ../fig/* ./fig
make html