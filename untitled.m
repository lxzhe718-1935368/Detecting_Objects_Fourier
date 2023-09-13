S = load('Kraken.mat');
for k = 1:size(S.Kraken,2);
    V = S.Kraken(:,k);
    F = sprintf('column_%d.mat',k);
    save(F,'V')
end