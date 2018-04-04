data=csvread('export.csv',1,1);
k_means_result=zeros(1,100);
for i = 1: 100
    [l,w]=size(data);
    if l<i
        break
    end
    [y,index,centroids]=Structure_prune(i);
    sil=silhouette(data,index);
    %disp(mean(sil));
    k_means_result(i)=mean(sil);
end
[M,I]=max(k_means_result);