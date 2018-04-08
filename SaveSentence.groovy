PrintWriter pw=new PrintWriter(new BufferedWriter(new FileWriter("experiments\\clean_train_tmp.csv")));
new File("train_1.csv").eachLine
{
	String[] f=it.split(",");
	if(f.length<6)
	println f[0]
	else
	pw.println(f[3].replace("\"","")+","+f[4].replace("\"","")+","+f[5].replace("\"",""));
}
pw.close();
