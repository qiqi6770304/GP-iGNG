function [V,net,genFlag] = TrainGrowingGasNet(V,temp1,net,scale,params,Global,wholeObj,genFlag,zmin)
%% Parameters
N = params.N;
MaxIt = params.MaxIt;
L = params.L;
epsilon_b = params.epsilon_b;
epsilon_n = params.epsilon_n;
alpha = params.alpha;
delta = params.delta;
T = params.T;
hab_threshold = params.hab_threshold;
insertion_threshold = params.insertion_threshold;
tau_b = params.tau_b;
tau_n = params.tau_n;


t = net.t;
w = net.w;
E = net.E;
C = net.C;
nx = net.nx;
ageSumBefore = net.ageSumBefore;
flag = net.flag;

% select the corner solutions using GNG
output=[];
fm =[];
C = net.C;
w= net.w;
for i = 1:size(w,1)
    neighbor = find(C(i,:)==1);
    %     [~,minInd] = min(w([i neighbor],:),[],1);
    %     if ~isempty(find(minInd==1))
    %         output = [output i];
    %     end
    %
    
    ageSum(i,:) = sum(t(i,neighbor,:),2);
    if ageSum(i,:) == ageSumBefore(i,:)
        flag(i,:) = flag(i,:) + 1;
    end
    
end

cw = w(output,:);
d = pdist2(cw,temp1);
[~,mdi] = min(d,[],2);

% temp1 = [temp1;repmat(temp1(mdi,:),1,1)];
% habn = net.habn;

% hold on
% PlotResults(w, C)
% update the net using the new generated reference vector 'temp'

% for ii = 1:size(w,1)
%     ageSum(ii,:) = sum(t(ii,find(C(ii,:) == 1),:),2);
%     if ageSum(ii,:) == ageSumBefore(ii,:)
%         flag(ii,:) = flag(ii,:) + 1;
%     end
% end
ageSumBefore = ageSum;
maxN = 1.5;

if Global.evaluated <= round(1.9*Global.evaluation)
    maxIter = 1;
    maxPZ = maxN;
    if size(w,1) == round(maxN*N)
        %      PlotResults(w, C)
        % plot(w(:,1),w(:,2),'k*')
        % hold on
        % plot(temp1(:,1),temp1(:,2),'rs')
        [~,rankFlag] = sort(flag,'descend');
        r = rankFlag(1:round(maxN*N)-N);
        %     [~,index] = max(flag,[],1);
        %     r = index;
        C(r, :) = [];
        C(:, r) = [];
        t(r, :) = [];
        t(:, r) = [];
        w(r, :) = [];
        E(r) = [];
        ageSumBefore(r,:) = [];
        flag(r,:) = [];
        flag = zeros(N,1);
        
    end
else
    if size(w,1) < round(maxN*N) && isempty(genFlag)
        maxPZ = maxN;
        maxIter = 1;
    else
        maxPZ = 1;
        maxIter = 0;
    end
    if size(w,1) == round(maxN*N)
        %         Choose = true(1,size(w,1));
        %         Cosine = 1 - pdist2(w,w,'cosine');
        %         Cosine(logical(eye(length(w)))) = 0;
        %         while sum(Choose) > Global.N
        %             Remain   = find(Choose);
        %             Temp     = sort(-Cosine(Remain,Remain),2);
        %             [~,Rank] = sortrows(Temp);
        %             Choose(Remain(Rank(1))) = false;
        %         end
        %         C(~Choose, :) = [];
        %         C(:, ~Choose) = [];
        %         t(~Choose, :) = [];
        %         t(:, ~Choose) = [];
        %         w(~Choose, :) = [];
        %         E(~Choose) = [];
        %         ageSumBefore(~Choose,:) = [];
        %         flag(~Choose,:) = [];
        %         flag = zeros(N,1);
        %
        %
        maxPZ = 1;
        maxIter = 0;
        genFlag = Global.gen;
    end
end
% plot(w(:,1),w(:,2),'k*')

if isempty(genFlag)
    % if Global.gen <= round(0.9*Global.maxgen)
    for iter = 1:maxIter
        for kk = 1:size(temp1,1)
            nx = nx + 1;
            x = temp1(kk,:);
            w = w./sum(w,2);
            d = pdist2(x, w);
            [~, SortOrder] = sort(d);
            s1 = SortOrder(1);
            s2 = SortOrder(2);
            
            % Aging: the age of all neighbours of s1 is increased by 1
            
            t(s1, :) = t(s1, :) + 1;
            t(:, s1) = t(:, s1) + 1;
            
            % Add Error
            E(s1) = E(s1) + d(s1)^2;
            
            % Adaptation
            %         if ~all(ismember(s1,output))
            w(s1,:) = w(s1,:) + epsilon_b*(x-w(s1,:));
            %         end
            %     habn(s1) = habn(s1) + habituate(habn(s1),tau_n)*habn(s1);
            Ns1 = find(C(s1,:)==1);
            for j=Ns1
                %             if ~all(ismember(j,output))
                w(j,:) = w(j,:) + epsilon_n*(x-w(j,:));
                %             end
                %         habn(j) = habn(j) + habituate(habn(j),tau_b)*habn(j);
            end
            
            % Create Link
            C(s1,s2) = 1;
            C(s2,s1) = 1;
            t(s1,s2) = 0;
            t(s2,s1) = 0;
            
            % Remove Old Links
            C(t>T) = 0;
            nNeighbor = sum(C);
            AloneNodes = (nNeighbor==0);
            if ~isempty(find(AloneNodes == true))
                %         AloneNodes
            end
            C(AloneNodes, :) = [];
            C(:, AloneNodes) = [];
            t(AloneNodes, :) = [];
            t(:, AloneNodes) = [];
            w(AloneNodes, :) = [];
            E(AloneNodes) = [];
            ageSumBefore(AloneNodes,:) = [];
            flag(AloneNodes,:) = [];
            %     habn(AloneNodes) = [];
            
            % Add New Nodes
            
            if mod(nx, L) == 0 && size(w,1) < round(maxPZ*N)
                %     bestDistance = exp(-pdist2(w(s1,:),x));
                %     if (habn(s1) < hab_threshold) & (bestDistance < insertion_threshold)
                [~, q] = max(E);
                [~, f] = max(C(:,q).*E);
                r = size(w,1) + 1;
                w(r,:) = (w(q,:) + w(f,:))/2;
                %         w(r,:) = (w(s1,:)+x)/2;
                C(q,f) = 0;
                C(f,q) = 0;
                C(q,r) = 1;
                C(r,q) = 1;
                C(r,f) = 1;
                C(f,r) = 1;
                t(r,:) = 0;
                t(:, r) = 0;
                E(q) = alpha*E(q);
                E(f) = alpha*E(f);
                E(r) = E(q);
                ageSumBefore(r,:) = 0;
                flag(r,:) = 0;
                %         habn(r) = 1;
            end
            
            % Decrease Errors
            E = delta*E;
        end
    end
    % % use distance upper bound to delete the useless nodes
    % d = pdist2(w, temp1);
    % [minD,~] = min(d,[],2);
    % % max(minD)
    % disBound = 0.2;
    % index = find(minD>disBound);
    % C(index, :) = [];
    % C(:, index) = [];
    % t(index, :) = [];
    % t(:, index) = [];
    % w(index, :) = [];
    % E(index) = [];
    % ageSumBefore(index,:) = [];
    % flag(index,:) = [];
    
    w = w./sum(w,2);
    net.w = w;
    net.E = E;
    net.C = C;
    net.t = t;
    net.nx = nx;
    net.ageSumBefore = ageSumBefore;
    net.flag = flag;
    % net.habn = habn;
end

if isempty(genFlag)
    output = [];
    for i = 1:size(w,1)
        neighbor = find(C(i,:)==1);
        [~,minInd] = min(w([i neighbor],:),[],1);
        [~,maxInd] = max(w([i neighbor],:),[],1);
        if ~isempty(find(minInd==1))
            output = [output i];
        elseif ~isempty(find(maxInd==1))
            output = [output i];
        end
        
    end
    
    for t = 1:size(output,2)
        s = output(:,t);
        x = mean(w(find(C(s,:)==1),:),1);
        w(s,:) = w(s,:) - 1*(x-w(s,:));
    end
    
    t = any(w<0,2);
    l = find(t == true);
    if ~isempty(l)
        for tt = 1:size(l,1)
            w(l(tt,1),any(w(l(tt,1),:)<0,1)) = 0;
        end
    end

    
    V = w;
    V = V.*repmat(scale,size(V,1),1);
end





if (Global.gen == genFlag) & 1==0
    N = size(wholeObj,1);
    %       (repmat(max(wholeObj,[],1),N,1) - repmat(min(wholeObj,[],1),N,1))
    wholeObj1 = (wholeObj - repmat(zmin,N,1));
    wholeObj = wholeObj1./scale;
%     temp2 = wholeObj./sum(wholeObj,2);
    temp2 = wholeObj1./sum(wholeObj1,2);
    
    M = size(wholeObj,2);
    H = [eye(M-1)-ones(M-1)/M;-ones(1,M-1)/M];
    Pe = H*inv(H'*H)*H';
    f = wholeObj*Pe';
    minf = min(f,[],1);
    maxf = max(f,[],1);
    f = (f-minf)./(maxf-minf);
    temp1 = f./sum(f,2);
    
    
    output = [];
    for i = 1:size(w,1)
        neighbor = find(C(i,:)==1);
        [~,minInd] = min(w([i neighbor],:),[],1);
        [~,maxInd] = max(w([i neighbor],:),[],1);
        if ~isempty(find(minInd==1))
            output = [output i];
        elseif ~isempty(find(maxInd==1))
            output = [output i];
        end
        
    end
    
    for t = 1:size(output,2)
        s = output(:,t);
        x = mean(w(find(C(s,:)==1),:),1);
        w(s,:) = w(s,:) - 1*(x-w(s,:));
    end
    
    t = any(w<0,2);
    l = find(t == true);
    if ~isempty(l)
        for tt = 1:size(l,1)
            w(l(tt,1),any(w(l(tt,1),:)<0,1)) = 0;
        end
    end
    
    temp1 = [temp2;net.w];
    
    %     dis =  pdist2(temp1,temp1);
    %     dis(logical(eye(length(dis)))) = inf;
    %     Choose = true(1,size(temp1,1));
    %
    %
    %     while sum(Choose) > Global.N
    %
    %         Remain   = find(Choose);
    %         [mindis,] = min(dis(Remain,Remain),[],2);
    %         [~,associate] = min(mindis,[],1);
    %         Choose(Remain(associate)) = false;
    %     end
    %
    %     V = temp1(Choose,:)./sum(temp1(Choose,:),2);
    %     V = V.*repmat(scale,size(V,1),1);
    
    Angle2 = acos(1-pdist2(temp1,temp1,'cosine'));
    Angle2(logical(eye(length(Angle2)))) = inf;
    [angle2,associate2] = min(Angle2,[],2);
    [~,maxAss2] = min(angle2,[],1);
    
    
    Choose = true(1,size(temp1,1));
    [~,Extreme1] = min(temp1,[],1);
    [~,Extreme2] = max(temp1,[],1);
    Choose(Extreme1) = false;
    Choose(Extreme2) = false;
    
    while sum(~Choose) < 1.5*params.N
        Remain   = find(Choose);
        [mindis,] = min(Angle2(Remain,setdiff(1:size(temp1,1),Remain)),[],2);
        [~,associate] = max(mindis,[],1);
        Choose(Remain(associate)) = false;
    end
    V = temp1(~Choose,:)./sum(temp1(~Choose,:),2);
    V = V.*repmat(scale,size(V,1),1);
    
        V = V.*repmat(scale,size(V,1),1);
    V = [w.*repmat(scale,size(w,1),1);temp2];
%     V = temp2;
end

end

