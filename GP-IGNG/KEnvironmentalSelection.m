function [index,net,genFlag,V] = KEnvironmentalSelection(PopObj,V,theta,A1,net,params,scale,Global,genFlag,zmin,w)
% The environmental selection of K-RVEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He
% if  w~=1
[FrontNo,MaxFNo] = NDSort(PopObj,min(size(PopObj,1),round(Global.N/Global.M)));
PopObjNo = PopObj(find(FrontNo<=MaxFNo),:);
[N,M] = size(PopObj);
NV    = size(V,1);

% else
%     PopObjNo = PopObj;
% end
%         PopObjNo = (PopObjNo - zmin)./(max(PopObjNo,[],1)-zmin);
%     PopObjNo = PopObjNo./sum(PopObjNo,2);
%     [~,ia,~] = unique(roundn(PopObjNo(:,1:M),-3),'rows','stable');
%     PopObjNo = PopObjNo(ia,:);
%
%     Num = min(size(PopObjNo,1), round(Global.N/Global.M));
%
% if size(PopObjNo,1) > Num
%     [IDX,C]   = kmeans(PopObjNo,Num);
%     wcid = [];
%     for i = 1:Num
%         id = find(IDX==i);
%         cid = id(randperm(size(id,1),1),:);
%         wcid = [wcid;cid];
%     end
%     PopObjNo =   PopObjNo(wcid,:);
%
% end


%     A11 = A1(NDSort(A1,1)==1);

% 	   Num = min(length(A1),size(PopObj,1));

%      A11 = A1(randperm(length(A1),Num));
A1obj = A1.objs;

A1objNo = (A1obj - min(A1obj,[],1))./scale;
A1objNo = A1objNo./sum(A1objNo,2);
[~,ia,~] = unique(roundn(A1objNo(:,1:M),-3),'rows','stable');
A1objNo = A1objNo(ia,:);
Num = min(size(A1objNo,1), round(Global.N/Global.M));

if length(A1)>1 && w~=1 && ~isempty(A1objNo) && ~isempty(Num) && size(A1objNo,1)>Num &&isempty(find(isnan(A1objNo)==true))
    
    
    %         Num = min(Num, size(PopObjNo,1));
    try
        [IDX,C]   = kmeans(A1objNo,Num);
        
        wcid = [];
        for i = 1:Num
            id = find(IDX==i);
            cid = id(randperm(size(id,1),1),:);
            wcid = [wcid;cid];
        end
        A1obj =   A1obj(wcid,:);
    catch e
%         A1objNo
%         Num
    end
end
wholeObj = [PopObjNo;A1obj];
%     zmin = min(wholeObj,[],1);
temp1 = (wholeObj- zmin)./scale;
temp1(size(PopObjNo,1)+1:end,:) = (A1obj - min(A1obj,[],1))./scale;
temp1 = temp1./sum(temp1,2);
[~,ia,~] = unique(roundn(temp1(:,1:M),-3),'rows','stable');
temp1 = temp1(ia,:);

temp1 = temp1(randperm(size(temp1,1)),:);

[~,id1] = min(temp1,[],1);
[~,id2] = max(temp1,[],1);

if size(temp1,1) > 2&&isempty(find(isnan(temp1)==true))&& Global.evaluated <= round(1*Global.evaluation) && isempty(genFlag)
    [V,net,genFlag] = TrainGrowingGasNet(V,temp1,net,scale,params,Global,[[];A1obj],genFlag,zmin);
end
Vb = V;
V = [V;temp1(id1,:).*scale;temp1(id2,:).*scale];
V = V./sum(V,2);
%     V = V(randperm(size(V,1),min(size(V,1),round(Global.N/Global.M))),:);
[~,ia,~] = unique(roundn(V(:,1:M),-3),'rows','stable');
V = V(ia,:);

%     plot(1:M,V,'g')
%% Translate the population
%     PopObj = PopObj - min(PopObj,[],1);
PopObj = PopObj - zmin;

%% Calculate the smallest angle value between each vector and others
cosine = 1 - pdist2(V,V,'cosine');
cosine(logical(eye(length(cosine)))) = 0;
gamma = min(acos(cosine),[],2);

%% Associate each solution to a reference vector
Angle = acos(1-pdist2(PopObj,V,'cosine'));
[~,associate] = min(Angle,[],2);

%% Select one solution for each reference vector
Next = zeros(1,NV);
for i = unique(associate)'
    current = find(associate==i);
    % Calculate the APD value of each solution
    APD = (1+M*theta*Angle(current,i)/gamma(i)).*sqrt(sum(PopObj(current,:).^2,2));
    % Select the one with the minimum APD value
    [~,best] = min(APD);
    Next(i)  = current(best);
end
% Population for next generation
index = Next(Next~=0);
V = Vb;
end