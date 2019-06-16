public class Playground {

    public static void main(String[] args) {
        int[] arr = new int[] {1,2,3,4,5};
        addOne(arr);
        System.out.println(arr[0]);
    }

    public static void addOne(int[] arr) {
        arr[0] ++;
    }

}
