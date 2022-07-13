function GPiGNG(Global)
% <algorithm> <G>
% Surrogate-assisted RVEA
% alpha ---  2 --- The parameter controlling the rate of change of penalty
% wmax  --- 20 --- Number of generations before updating Kriging models
% mu    ---  5 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------
% T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya, A surrogate-
% assisted reference vector guided evolutionary algorithm for
% computationally expensive many-objective optimization, IEEE Transactions
% on Evolutionary Computation, 2018, 22(1): 129-142.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

%% Parameter setting
rng('shuffle');
rng(randperm(2^32-1,1),'twister');


[alpha,wmax,mu] = Global.ParameterSet(2,20,5);

params.N = round(Global.N/Global.M);
params.MaxIt = 20;
params.L = params.N;
params.epsilon_b = 0.2;
params.epsilon_n = 0.006;
params.alpha = 0.5;
params.delta = 0.995;
params.T =round(params.N*0.2);
params.hab_threshold = 0.1;
params.insertion_threshold = 0.95;
params.tau_b = 0.3;
params.tau_n = 0.1;

%% Generate the reference points and population
[V0,Global.N] = UniformPoint(Global.N,Global.M);
V     = V0;
NI    = 11*Global.D-1;
P     = lhsamp(NI,Global.D);
A2    = INDIVIDUAL(repmat(Global.upper-Global.lower,NI,1).*P+repmat(Global.lower,NI,1));
A1    = A2;
THETA = 5.*ones(Global.M,Global.D);
Model = cell(1,Global.M);
count =[];
zmin       = min(A2.objs,[],1);

net = InitilizeGrowingGasNet(V,A1,params);
genFlag =[];
scale = ones(1,Global.M);
New = [];
ccTimes = 1;
Runtime = zeros(455,2);
Runtime(:,1) = Global.run;
Runtime(Global.evaluated,2) = Global.runtime;
% filefolder = ['Data\Archive',num2str(ccTimes)];
% save(filefolder, 'A2')
%% Optimization
while Global.NotTermination(A2)
    % Refresh the model and generate promising solutions
    A1Dec = A1.decs;
    A1Obj = A1.objs;

    A1(find(sum(isnan(A1Obj),2)>0))=[];
    A1Dec(find(sum(isnan(A1Obj),2)>0),:)=[];
    A1Obj(find(sum(isnan(A1Obj),2)>0),:)=[];

    A1(find(sum(isinf(A1Obj),2)>0))=[];
    A1Dec(find(sum(isinf(A1Obj),2)>0),:)=[];
    A1Obj(find(sum(isinf(A1Obj),2)>0),:)=[];

    [c,ia,ic] = unique(A1Obj,'rows');
    A1Obj = A1Obj(ia,:);
    A1Dec = A1Dec(ia,:);
    A1 = A1(ia);
    [c,ib,ic] = unique(A1Dec,'rows');
    A1Obj = A1Obj(ib,:);
    A1Dec = A1Dec(ib,:);
    A1 = A1(ib);

    A2Obj = A2.objs;
    A2Dec = A2.decs;

    zmin = min(A2Obj,[],1);
    for i = 1 : Global.M
        % The parameter 'regpoly1' refers to one-order polynomial
        % function, and 'regpoly0' refers to constant function. The
        % former function has better fitting performance but lower
        % efficiency than the latter one

        [c,ia,ic] = unique(A1Obj,'rows','stable');
        A1Obj = A1Obj(ia,:);
        A1Dec = A1Dec(ia,:);
        A1 = A1(ia);
        try
            dmodel     = dacefit(A1Dec,A1Obj(:,i),'regpoly0','corrgauss',THETA(i,:),1e-5.*ones(1,Global.D),100.*ones(1,Global.D));
            Model{i}   = dmodel;
            THETA(i,:) = dmodel.theta;
        catch exception
            A1Obj(:,i)
        end
    end

    w      = 1;

    [FrontNo,MaxFNo] = NDSort(A2.objs,min(size(A2.objs,1),Global.N));
    A2_ = A2(find(FrontNo<=MaxFNo));
    A22 = A2(find(FrontNo==1));

    PopDec = A1.decs;



    [FrontNo,MaxFNo] = NDSort(A1.objs,A1.cons,length(A1));
    A11 = A1(find(FrontNo==1));

    if  isempty(genFlag)
        net = InitilizeGrowingGasNet(V0,A22,params);
        V = net.w;
    end
    WPopDec = [];
    WPopObj = [];
    WMSE = [];

    while w <= wmax
        drawnow();
        OffDec = GA(PopDec);
        PopDec = [PopDec;OffDec];
        [N,~]  = size(PopDec);
        PopObj = zeros(N,Global.M);
        MSE    = zeros(N,Global.M);
        for i = 1: N
            for j = 1 : Global.M
                [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),Model{j});
            end
        end

        zmin       = min([zmin;PopObj],[],1);

        [index,net,genFlag,V]  = KEnvironmentalSelection(PopObj,V,(w/wmax)^alpha,A22,net,params,scale,Global,genFlag,zmin,w);
        PopDec = PopDec(index,:);
        PopObj = PopObj(index,:);
        MSE = MSE(index,:);

        WPopDec = [WPopDec;PopDec];
        WPopObj = [WPopObj;PopObj];
        WMSE = [WMSE;MSE];
        % Adapt referece vectors
        if ~mod(w,ceil(wmax*0.1)) &&  size(PopObj,1)>2

            scale = max([A22.objs;PopObj],[],1)-min([A22.objs;PopObj],[],1);

        end
        w = w + 1;
    end

    % Select mu solutions for re-evaluation
%     [NumVf,~] = NoActive(A1Obj,V0);
    NumVf =[];

    [~,ib]= intersect(WPopDec,A2.decs,'rows');
    WPopObj(ib,:)=[];
    WPopDec(ib,:)=[];
    WMSE(ib,:)=[];

    [c,ia,ic] = unique(WPopObj,'rows');
    if ~isempty(ia)
        WPopObj = WPopObj(ia,:);
        WPopDec = WPopDec(ia,:);
        WMSE = WMSE(ia,:);
    end

    [~,ib]= intersect(WPopDec,A2.decs,'rows');
    WPopObj(ib,:)=[];
    WPopDec(ib,:)=[];
    WMSE(ib,:)=[];

    [FrontNo,MaxFNo] = NDSort(WPopObj,size(WPopObj,1));
    MaxFNo =  1;
    WPopObj = WPopObj(find(FrontNo<=MaxFNo),:);
    WPopDec = WPopDec(find(FrontNo<=MaxFNo),:);
    WMSE = WMSE(find(FrontNo<=MaxFNo),:);



    if ~isempty(PopObj)
        PopObj = WPopObj;
        PopDec = WPopDec;
        MSE = WMSE;
        [PopNew,count]    = KrigingSelect6(PopDec,PopObj,MSE,V,V0,NumVf,0.05*Global.N,mu,(w/wmax)^alpha,count,zmin,A22);

        if ~isempty(PopNew)
            [~,ib]= intersect(PopNew,A2.decs,'rows');
            PopNew(ib,:) = [];
        end

        if ~isempty(PopNew)
            New       = INDIVIDUAL(PopNew);
            zmin       = min([zmin;New.objs],[],1);

            A2        = [A2,New];
            theta2 = (Global.evaluated./Global.evaluation)^alpha;
            A1        = UpdataArchive(A1,New,V,mu,NI,A2,theta2);
        else
            New = [];
        end

    end
    A1        = A2;

    A2(find(sum(isnan(A2.objs),2)>0))=[];
    A2(find(sum(isinf(A2.objs),2)>0))=[];


end
end