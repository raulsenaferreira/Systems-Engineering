#include <iostream>
#include <string>
using namespace std;

void imprimePilha(char vet[], int size){
	for (int i = 0; i < size; i++){
		cout << vet[i];
	}
	cout << vet[size] << endl;
}

int main(int argc, char const *argv[]){
	string expressao;
	char pilhaNum[200];
	char pilhaOp[200];

	while(getline(cin, expressao)){
		int i = 0;
		int contP=0;
		int contN=0;
		int topoNum=0;
		int topoOp=0;

		while(expressao[i]!='\0'){
			if(expressao[i]=='+' || expressao[i]=='-' || expressao[i]=='*' || expressao[i]=='/'){
				pilhaOp[topoOp++]=expressao[i];
			}
			else if (expressao[i]=='('){
				contP++;
				contN=0;
			}
			else if(expressao[i]==')'){
				pilhaNum[topoNum++]=pilhaOp[--topoOp];
				contP--;
			}
			else{
				pilhaNum[topoNum++]=expressao[i];
				contN++;
			}

			if(expressao[i]=='\0'){

				while(topoOp!=0){
					pilhaNum[topoNum++]=pilhaOp[--topoOp];
				}
			}
			i++;
		}

		imprimePilha(pilhaNum, topoNum);
	}
	return 0;
}