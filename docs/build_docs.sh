PATH=../:$PATH
mkdir -p ./notebooks
cp ../notebooks/* ./notebooks
mkdir -p ./fig
cp ../fig/* ./fig
make html