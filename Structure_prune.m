function [y, Idx, Ctrs]=Structure_prune(k)
data=csvread('export.csv',1,1);
[Idx,Ctrs] = kmeans(data,k);
closestIdx=zeros(1,k,'uint16');
for iCluster = 1:max(Idx)
    %# find the points that are part of the current cluster
    currentPointIdx = find(Idx==iCluster);
    %# find the index (among points in the cluster)
    %# of the point that has the smallest Euclidean distance from the centroid
    %# bsxfun subtracts coordinates, then you sum the squares of
    %# the distance vectors, then you take the minimum
    [~,minIdx] = min(sum(bsxfun(@minus,data(currentPointIdx,:),Ctrs(iCluster,:)).^2,2));
    %# store the index into X (among all the points)
    closestIdx(iCluster) = currentPointIdx(minIdx);
end
y=closestIdx;
end
