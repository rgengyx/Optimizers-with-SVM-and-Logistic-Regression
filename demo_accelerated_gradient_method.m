% Add files to path
addpath(genpath('method'));

% Global Seed Settings
rng("default");


%%%%%%%%%%%%%%%%%%%
% Declare Options %
%%%%%%%%%%%%%%%%%%%

% AGM
opts.agm.maxit = 20000;
opts.agm.tol = 1e-4;
opts.agm.print = true;


%%%%%%%%%%%%%%%%%%%%
% Define Variables %
%%%%%%%%%%%%%%%%%%%%

m = 300;
n = 3000;
s = 30;
mu = 1;
A = randn(m,n);
delta = 1e-3;
nu = 1e-4;

mask=randperm(n,s);
x_star = zeros(n,1);
x_star(mask) = randn(s,1);
b=A*x_star+ 0.01*randn(m,1);

% Define Step size
L1 = 2 * mu + norm(A'*A);
L2 = mu * 1/delta + norm(A'*A);

%%%%%%%%%%%%%%%%%%%
% Declare Options %
%%%%%%%%%%%%%%%%%%%

f.obj = @obj;
f.grad = @grad;
opts.agm.mu = mu;
opts.agm.A = A;
opts.agm.b = b;
opts.agm.delta = delta;
opts.agm.nu = nu;
opts.agm.beta = @beta;
opts.agm.L1 = L1;
opts.agm.L2 = L2;

% Line Search for Unknown Lipschitz Constant
opts.ls.beta = 0.8;
opts.ls.l = 0.8;

x0 = zeros(n,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call Optimization Methods %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% phis = ["phi1" "phi2"];
phis = ["phi1" "phi2" "phi3"];
kss = {};
ngss = {};
xs = {};

for i=1:length(phis)
    opts.agm.phi = phis{i};
    opts.agm.step_size = step_size(opts);
    [x,ks,ngs] = accelerated_gradient_method(f,x0,opts);
%     [x,ks,ngs] = inertial_gradient_method(f,x0,opts);
    kss{i} = ks;
    ngss{i} = ngs;
    xs{i} = x;
end

%%%%%%%%%%%%%%%%%%%%
% Convergence plot %
%%%%%%%%%%%%%%%%%%%%

for i=1:length(phis)
    figure("Name", "Convergence Plot:"+phis{i});
    plot(kss{i},log(ngss{i}),'Color',"red");
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstructed solutions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:length(phis)
    figure("Name", "Reconstruction Plot:"+phis{i});
    scatter(1:n,xs{i}');
    hold on
    yline(0);
    hold off
end

% Compare with x star
x = linspace(-5,5,500);
y = x;
for i=1:length(phis)
    figure("Name", "Compare wtih x* Plot:"+phis{i});
    scatter(x_star',xs{i}');
    hold on
    plot(x,y);
    hold off
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Utility Functions Declarations %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define Phi
function phi1_x = phi1(x)
    phi1_x = norm(x)^2;
end

function phi2_x = phi2(x,opts)
    delta = opts.agm.delta;
    phi2_x = 0;
    for i=1:length(x)
        t = x(i);
        if abs(t) <= delta 
            phi_hub = 1/(2*delta)*t^2;
        else
            phi_hub = abs(t) - 0.5*delta;
        end
        phi2_x = phi2_x + phi_hub;
    end
end

function phi3_x = phi3(x,opts)
    nu = opts.agm.nu;
    phi3_x = 0;
    for i=1:length(x)
        t = x(i);
        phi3_x = phi3_x + log(1+t^2/nu);
    end
end

% Define Phi_grad
function phi1_x_grad = phi1_grad(x)
    phi1_x_grad = 2*x;
end

function phi2_x_grad = phi2_grad(x,opts)
    delta = opts.agm.delta;
    phi2_x_grad = [];
    for i=1:length(x)
        t = x(i);
        if abs(t) <= delta
            phi_hub_grad = 1/delta * t;
        else 
            phi_hub_grad = t/abs(t);
        end
        phi2_x_grad = [phi2_x_grad;phi_hub_grad];
    end
end

function phi3_x_grad = phi3_grad(x,opts)
    nu = opts.agm.nu;
    phi3_x_grad = [];
    for i=1:length(x)
        t = x(i);
        phi3_x_grad = [phi3_x_grad;(2*t/nu) / (1 + t^2/nu)];
    end
end

% Define objective function
function fx = obj(x, opts)
    if opts.agm.phi == "phi1"
        phi = phi1(x);
    elseif opts.agm.phi == "phi2"
        phi = phi2(x,opts);
    elseif opts.agm.phi == "phi3"
        phi = phi3(x,opts);
    end
    A = opts.agm.A;
    mu = opts.agm.mu;
    b = opts.agm.b;
    fx = 0.5*norm(A*x-b)^2+mu*phi;
end

function gx = grad(x, opts)
    if opts.agm.phi == "phi1"
        phi_grad = phi1_grad(x);
    elseif opts.agm.phi == "phi2"
        phi_grad = phi2_grad(x,opts);
    elseif opts.agm.phi == "phi3"
        phi_grad = phi3_grad(x,opts);
    end
    A = opts.agm.A;
    b = opts.agm.b;
    gx = A'*(A*x-b) + phi_grad;
end

% Define Extrapolation parameter beta
function [prev_t, beta] = beta(prev_t)
    t = (1/2)*(1 + sqrt(1+4*prev_t^2));
    beta = (prev_t - 1) / t;
    prev_t = t;
end

% Step size
function step_size = step_size(opts)
    L1 = opts.agm.L1;
    L2 = opts.agm.L2;
    if opts.agm.phi == "phi1"
        step_size = 1 / L1;
    else
        step_size = 1 / L2;
    end
end