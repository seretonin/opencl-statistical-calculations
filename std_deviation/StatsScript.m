clear;

filename = '256_23.csv';
M = csvread(filename);

s = sum(M)
mode = mode(M);
median = median(M);
mean = mean(M)
max = max(M);
min = min(M);
std_deviation = std(M)
var = var(M);