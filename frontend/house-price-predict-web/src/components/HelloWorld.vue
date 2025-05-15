<template>
  <form @submit.prevent="predict" class="predict-form">
    <input v-model.number="form.square" placeholder="面积（㎡）" required>
    <input v-model.number="form.total_floor" placeholder="总楼层" required>

    <select v-model="form.floor_level" required>
      <option disabled value="">请选择楼层</option>
      <option value="低">低楼层</option>
      <option value="中">中楼层</option>
      <option value="高">高楼层</option>
    </select>

    <select v-model.number="form.room" required>
      <option disabled value="">请选择几室</option>
      <option v-for="n in 6" :key="n" :value="n">{{ n }}室</option>
    </select>

    <select v-model.number="form.hall" required>
      <option disabled value="">请选择几厅</option>
      <option v-for="n in 3" :key="n" :value="n">{{ n }}厅</option>
    </select>

    <select v-model.number="form.kitchen" required>
      <option disabled value="">请选择几厨</option>
      <option v-for="n in 2" :key="n" :value="n">{{ n }}厨</option>
    </select>

    <select v-model.number="form.bath" required>
      <option disabled value="">请选择几卫</option>
      <option v-for="n in 2" :key="n" :value="n">{{ n }}卫</option>
    </select>

    <select v-model="form.direction_simple" required>
      <option disabled value="">请选择朝向</option>
      <option value="南北">南北</option>
      <option value="南">南</option>
      <option value="北">北</option>
      <option value="东">东</option>
      <option value="西">西</option>
    </select>

    <select v-model="form.decoration" required>
      <option disabled value="">请选择装修</option>
      <option value="简装">简装</option>
      <option value="精装">精装</option>
      <option value="毛坯">毛坯</option>
    </select>

    <select v-model="form.elevator" required>
      <option disabled value="">有无电梯</option>
      <option value="有">有</option>
      <option value="无">无</option>
    </select>

    <select v-model="form.ownership" required>
      <option disabled value="">请选择产权</option>
      <option value="商品房">商品房</option>
      <option value="经济适用房">经济适用房</option>
      <option value="公房">公房</option>
    </select>

    <button type="submit" :disabled="loading">
      {{ loading ? '预测中...' : '预测房价' }}
    </button>

    <div v-if="error" class="error">❌ {{ error }}</div>
    <div v-if="result" class="result">✅ 预测结果：{{ result }} 元/㎡</div>
  </form>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const form = ref({
  square: '',
  total_floor: '',
  floor_level: '',
  room: '',
  hall: '',
  kitchen: '',
  bath: '',
  direction_simple: '',
  decoration: '',
  elevator: '',
  ownership: ''
})
const result = ref(null)
const error = ref(null)
const loading = ref(false)

const predict = async () => {
  result.value = null
  error.value = null
  loading.value = true

  try {
    const res = await axios.post('http://127.0.0.1:5000/predict', form.value)
    result.value = res.data.predicted_price
  } catch (err) {
    error.value = err.response?.data?.message || '预测失败，请检查输入或稍后重试。'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.predict-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: 360px;
  margin: auto;
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: #f9f9f9;
}
input, select, button {
  padding: 8px;
  font-size: 1rem;
}
.result {
  color: green;
}
.error {
  color: red;
}
</style>
