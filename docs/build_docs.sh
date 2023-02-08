PATH=../:$PATH

echo "Copying and executing jupyter notebooks"
mkdir -p ./notebooks
cp ../notebooks/*.ipynb ./notebooks

cd notebooks || return
jupyter run ./*.ipynb
cd ../

echo "Copying figures"
mkdir -p ./fig
cp ../fig/* ./fig

echo "Making docs"
make html