#ifndef CLIENT_H
#define CLIENT_H

#include <iostream>
using namespace std;

class Client
{
protected:
    string m_name;
    string m_adress;
    string m_job;
    float m_income;

public:
    Client(string name, string adress, string job, float income) : m_name(name), m_adress(adress), m_job(job), m_income(income){};
    ~Client(){};

    string Name() {return m_name; }
    string Adress() {return m_adress; }
    string Job() {return m_job; }
    float Income() { return m_income; }
};

#endif // CLIENT_H
