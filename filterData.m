function filterData(foldername, numOfFiles)
N = numOfFiles;

prefix = strcat(foldername, '/grip');

d = fdesign.lowpass('Fp,Fst,Ap,Ast',.1,1,0.5,40,200);
Hd = design(d,'FIR');

for i=0:N-1
    path = strcat(prefix, num2str(i));
    outpath = strcat(foldername, strcat('/filt', strcat(num2str(i), '.csv')));
    path = strcat(path, '.csv');
    tempdata = importFile(path);
    nn_data = filter(Hd, tempdata);
    csvwrite(outpath, nn_data);
end




        

    