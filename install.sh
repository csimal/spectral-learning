
echo "Installing Julia"

wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz
tar zxvf julia-1.7.2-linux-x86_64.tar.gz

echo "Don't forget to add the path to the julia executable to your path."

echo "See https://julialang.org/downloads/platform/ for details"

echo "Installing Julia Packages"

./julia-1.7.2/bin/julia packages.jl