function [PopNew,count]  = KrigingSelect6(PopDec,PopObj,MSE,V,V0,NumV1,delta,mu,theta,count,zmin,A1)
% Kriging selection in K-RVEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

A1Obj= A1.objs;
mu = min(mu,size(A1Obj,1));
LCB = PopObj;


MSE = max(0,MSE);
kk = rand(1,size(A1Obj,2));
S = sqrt(max(0,MSE));

S = MSE.*(MSE<=1)+S.*(MSE>1);

LCB_minus= PopObj-kk.*sqrt( S);

UCB_plus= PopObj+kk.*sqrt( S);
whole = [LCB;UCB_plus;A1Obj];
wholeOO = whole;
Zmin = min([whole;zmin],[],1);
zmin = min([whole;[]],[],1);
T=2;


scale = (max(whole,[],1)-min(Zmin,[],1));
%  scale = (max(A1Obj,[],1)-min(A1Obj,[],1));
zeroid= find(scale==0);
scale(:,zeroid) =10^(-6);
wholeNN= whole-min(whole,[],1);
whole=(whole-min(Zmin,[],1))./scale;

archive_index = [T*size(LCB,1)+1:size(whole,1)];
next = [];
mu_o = mu;
mu = 1*mu;
if size(whole,1) > mu

    [IDX,C]   = kmeans(whole,mu);


    [FrontNo,MaxFNo] = NDSort(C,mu);
    for center = 1:size(C,1)
        if FrontNo(center)==1
            %                 if 1==1
            s= find(IDX==center);

            evaluatedSol = intersect(s,archive_index);
            noEvaluatedSol = setdiff(s,evaluatedSol);

            if ~isempty(noEvaluatedSol)
                %     angel
                angleRank = [];
                if ~isempty(evaluatedSol)
                    Angle = acos(1-pdist2(whole(noEvaluatedSol,:),whole(evaluatedSol,:),'cosine'));
                else
                    Angle = acos(1-pdist2(whole(noEvaluatedSol,:),whole(archive_index,:),'cosine'));
                end
                [minAngle,~] = min(Angle,[],2);
                [~,angleRank] = sort(minAngle,'descend');

                RC = [];
                if ~isempty(evaluatedSol) && 1==0

                    Angle = acos(1-pdist2(whole(evaluatedSol,:),whole(evaluatedSol,:),'cosine'));
                    Angle(logical(eye(size(evaluatedSol,1))))=inf;
                    [minAngle,~] = min(Angle,[],2);
                    [~,id] = max(minAngle,[],1);
                    RC = evaluatedSol(id,1);
                    RC = evaluatedSol(randperm(size(evaluatedSol,1),1),1);
                end

                % distance

                cd = [];
                wholeNN = wholeOO;
                for i = 1:size(noEvaluatedSol,1)
                    A2Obj = wholeNN(evaluatedSol,:);
                    if ~isempty(A2Obj)
                        d=  pdist2(wholeNN(noEvaluatedSol(i,:),:),wholeNN(evaluatedSol,:),'euclidean');
                        A = wholeNN(noEvaluatedSol(i,:),:);
                        [cd_,id] = sort(d,2);
                        id_ = id(:,1);
                        k = any(A<A2Obj(id_,:),2) - any(A>A2Obj(id_,:),2);
                        cd_(find(k~=1),1)=0;

                        cd(i,:) = sum(cd_(:,1));
                    else
                        cd(i,:) = 0;
                    end
                end
                %                         [~,cbest] = max(cd,[],1);
                [~,disRank] = sort(cd,'descend');
                cc = find(cd==0);

                if size(cc,1)>1
                    %                                                         d_cc = pdist2(wholeNN(cc,:),zeros(1,size(LCB,2)),'euclidean');
                    d_cc = pdist2(wholeOO(cc,:),Zmin,'euclidean');
                    [~,disRank1] = sort(d_cc,'ascend');

                    if size(cc,1) > size(noEvaluatedSol,1)
                        size(noEvaluatedSol,1)-size(cc,1)+1:size(noEvaluatedSol,1)
                    end


                    %                             disRank(size(noEvaluatedSol,1)-size(cc,1)+1:size(noEvaluatedSol,1),:)
                    %                             cc(disRank1)
                    disRank(size(noEvaluatedSol,1)-size(cc,1)+1:size(noEvaluatedSol,1),:) = cc(disRank1);
                end


                sumRank = [];

                for t = 1:size(noEvaluatedSol,1)
                    if ~isempty(angleRank)
                        if size(cc,1)==size(noEvaluatedSol,1)
                            sumRank(t,:) =find(angleRank == t) +  find(disRank == t);
                        else
                            sumRank(t,:) = find(disRank == t);

                        end
                        sumRank(t,:) = find(angleRank == t) +  find(disRank == t);
                    else
                        sumRank(t,:) = find(disRank == t);
                    end
                end

                [~,best]    = min(sumRank);

                if noEvaluatedSol(best,:) > 2*size(LCB,1)
                    BB = noEvaluatedSol(best,:) -  2*size(LCB,1);
                    %                     BB = [];
                elseif noEvaluatedSol(best,:) > size(LCB,1)
                    BB = noEvaluatedSol(best,:) -  1*size(LCB,1);
                    %                     BB = [];
                else
                    BB = noEvaluatedSol(best,:);
                end
            else

                BB = [];

            end
            next =  [next BB];
        end
    end
    next = unique(next);
    if size(next,2)>mu_o
        next =next(randperm(mu_o));
    end
    PopNew = PopDec(next,:);
end
if isempty(next)
    PopNew = [];
end
%     hold on
%     plot(wholeB(next,1),wholeB(next,2),'ro')
end


