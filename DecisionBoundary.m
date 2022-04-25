clear
close all
clc
numberofdata=100;

mu1=[2,3];
sigma1=[1,1.5;1.5,3];
rng default;

rndnumber1=mvnrnd(mu1,sigma1,numberofdata);

mu2=[5,4];
sigma2=[1,1.5;1.5,3];
rng default;

rndnumber2=mvnrnd(mu2,sigma2,numberofdata);

figure; plot(rndnumber1(:,1),rndnumber1(:,2),'+');
hold on
plot(rndnumber2(:,1),rndnumber2(:,2),'o');


x0=ones(2*length(rndnumber1(:,1)),1);

x1=[rndnumber1(:,1);rndnumber2(:,1)];
x2=[rndnumber1(:,2);rndnumber2(:,2)];

X=[x0 mat2gray(x1) mat2gray(x2)];
y=[zeros(length(rndnumber1(:,1)),1);ones(length(rndnumber1(:,1)),1)];

theta=[0,0,0]';
alpha=1;
m = length(y); % number of training examples
h=sigmoid(X*theta);
[J] = costFunction(h, y,m);
epsilon=1e-3;


i=1;
while J>epsilon
    h=sigmoid(X*theta);
    theta=GT(theta,m,alpha,X,h,y);
    J= costFunction(h,y,m);
    i=i+1;
end

figure
a=find(y==1); b=find(y==0);
plot(X(a,2),X(a,3),'+')
hold on
plot(X(b,2),X(b,3),'o')
hold on
plot_x=[min(X(:,2)) max(X(:,2))];
plot_y=-1/theta(3)*(theta(1)+theta(2)*plot_x);
plot(plot_x,plot_y)
legend('pozitif', 'negatif','DecisionBoundry')
hold off




function g = sigmoid(z)

g=1./(1+exp(-z));

end

function [J] = costFunction(h,y,m)

J=1/m*(-y'*log(h)-(1-y)'*log(1-h));

end

function theta= GT(theta,m,alpha,X,h,y)

grad=1/m*(X'*(h-y)); % X has size (m x n) and (h-y) has size (m x 1)
theta=theta-alpha*grad;

end
