BufferedReader br=new BufferedReader(new FileReader(".\\experiments\\clean_train_tmp.csv"));
    String rd="";
    while(rd!=null)
    {
        if(!rd.equals(""))
        {
        String[] s1=rd.split(",");
        System.out.println(s1[0]+s1[1]+Integer.parseInt(s1[2]));
    }
        rd=br.readLine();
        }
