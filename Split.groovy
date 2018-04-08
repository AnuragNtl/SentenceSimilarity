static void main(String[] args)
{
	def l1=0;
	if(args.length==0)
{
	args=new String[1];
	args[0]="-1";
}
def f1=new PrintWriter(new BufferedWriter(new FileWriter("experiments\\train1_tmp.csv")));
def f2=new PrintWriter(new BufferedWriter(new FileWriter("experiments\\tests1_tmp.csv")));
def ct=0;
l1=Integer.parseInt(args[0]);
new File("experiments\\clean_train_tmp.csv").eachLine
{
	line->
	if(ct<l1)
f1.println(line);
else
f2.println(line);
ct++;
}
println ct;
f1.close();
f2.close();
}
