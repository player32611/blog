<template>
    <div id="sakana-widget"></div>
</template>

<script>
    import 'sakana-widget/lib/index.css';
    import SakanaWidget from 'sakana-widget';
    new SakanaWidget().mount('#sakana-widget');
    export default {
        name:"SakanaWidget",
        data(){
            return{}
        }
    }
</script>

<style scoped>

</style>